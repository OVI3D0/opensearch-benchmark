# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""
Runners for ClickHouse. They use the native clickhouse-connect client, so workloads send SQL rather than OpenSearch
query DSL: search operations have a body of the form ``{"sql": "...", "parameters": {...}, "settings": {...}}`` and
indices are tables whose body contains the ``CREATE TABLE`` statement as ``ddl``.
"""

import json
import logging

from osbenchmark import exceptions, workload
from osbenchmark.database.clients.clickhouse.helpers import elapsed_ms, parse_bulk_body, quote_identifier, to_python
from osbenchmark.worker_coordinator.runner import Retry, Runner, request_context_holder

logger = logging.getLogger(__name__)


def request_errors():
    """:return: The exceptions that mean a ClickHouse request failed, as opposed to a bug in OSB."""
    # pylint: disable=import-outside-toplevel
    import aiohttp
    from clickhouse_connect.driver import exceptions as clickhouse_exceptions
    return clickhouse_exceptions.Error, aiohttp.ClientError, OSError


def failure(weight, unit, e):
    return {"weight": weight, "unit": unit, "success": False, "error-type": "clickhouse", "error-description": str(e)}


async def timed(request):
    """Awaits a ClickHouse request and records it as the request for OSB's service time."""
    request_context_holder.on_request_start()
    try:
        return await request
    finally:
        request_context_holder.on_request_end()


def table_names(params):
    indices = params.get("indices")
    if indices:
        return [name for entry in indices for name in str(entry).split(",") if name.strip()]
    index = params.get("index") or params.get("table")
    if not index:
        raise exceptions.DataError("The operation needs an 'index' parameter with the name of the ClickHouse table.")
    return [name.strip() for name in index.split(",") if name.strip()]


ALL_TABLES = "_all"

MERGE_TREE_TABLES = "SELECT name FROM system.tables WHERE database = currentDatabase() AND engine LIKE '%MergeTree'"


async def resolve_tables(client, tables):
    """:return: The table names, with ``_all`` replaced by every MergeTree table of the current database."""
    if ALL_TABLES not in tables:
        return tables
    result = await client.query(MERGE_TREE_TABLES)
    return [row[0] for row in result.result_rows]


def sql_body(params, runner):
    body = params.get("body") or {}
    if "sql" not in body:
        raise exceptions.DataError(
            f"Operation [{runner}] needs a body with an 'sql' key. ClickHouse workloads send SQL, not OpenSearch query DSL.")
    return body["sql"], to_python(body.get("parameters")), body.get("settings")


class ClickHouseBulk(Runner):
    """
    Inserts the documents of a bulk body. With ``column-names``, documents are inserted as rows of those columns;
    otherwise they are inserted as JSON (``JSONEachRow``), so their keys must match the table's columns.

    ClickHouse inserts are atomic: a failed bulk request counts as one error.
    """

    async def __call__(self, clickhouse, params):
        table = table_names(params)[0]
        bulk_size = params.get("bulk-size", 0)
        unit = params.get("unit", "docs")
        columns = params.get("column-names")
        request_context_holder.on_client_request_start()
        try:
            client = await clickhouse.native()
            body = params["body"]
            if columns:
                rows = [tuple(to_python(doc.get(column)) for column in columns) for doc in parse_bulk_body(body)]
                request = client.insert(table, rows, column_names=columns, settings=params.get("settings"))
            else:
                request = client.raw_insert(table, insert_block=json_lines(body), fmt="JSONEachRow", settings=params.get("settings"))
            summary = await timed(request)
        except request_errors() as e:
            self.logger.warning("ClickHouse bulk insert into [%s] failed: %s", table, e)
            return {**failure(bulk_size, unit, e), "error-count": 1}
        finally:
            request_context_holder.on_client_request_end()
        return {"weight": bulk_size, "unit": unit, "success": True, "error-count": 0,
                "took": elapsed_ms(getattr(summary, "summary", None))}

    def __repr__(self, *args, **kwargs):
        return "clickhouse-bulk"


def json_lines(body):
    """
    :param body: A bulk body.
    :return: The documents of the bulk body as newline-delimited JSON. Newline-delimited bodies are passed through
             without decoding the documents, which keeps the load generator's CPU cost low.
    """
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")
    if isinstance(body, str):
        lines = [line for line in body.split("\n") if line.strip()]
        if len(lines) % 2 != 0:
            raise exceptions.BenchmarkError("The bulk body has an action without a document.")
        parse_bulk_body("\n".join(f"{action}\n{{}}" for action in lines[0::2]))   # validates the actions only
        return ("\n".join(lines[1::2]) + "\n").encode("utf-8")
    return "".join(json.dumps(to_python(doc), separators=(",", ":")) + "\n" for doc in parse_bulk_body(body)).encode("utf-8")


class ClickHouseQuery(Runner):
    """Runs the SQL query in the body. Reports the number of returned rows as hits."""

    async def __call__(self, clickhouse, params):
        sql, parameters, settings = sql_body(params, self)
        request_context_holder.on_client_request_start()
        try:
            client = await clickhouse.native()
            result = await timed(client.query(sql, parameters=parameters, settings=settings))
        except request_errors() as e:
            self.logger.warning("ClickHouse query failed: %s", e)
            return failure(1, "ops", e)
        finally:
            request_context_holder.on_client_request_end()
        return {"weight": 1, "unit": "ops", "success": True, "hits": result.row_count, "hits_relation": "eq",
                "timed_out": False, "took": elapsed_ms(result.summary)}

    def __repr__(self, *args, **kwargs):
        return "clickhouse-query"


class ClickHouseScrollQuery(Runner):
    """
    Pages through the results of the SQL query in the body. The SQL must page with ``{limit:UInt32}`` and
    ``{offset:UInt32}`` parameters and order by a deterministic key. Reads ``pages`` pages of ``results-per-page`` rows
    and stops early at the last page.
    """

    async def __call__(self, clickhouse, params):
        sql, parameters, settings = sql_body(params, self)
        if "{limit:" not in sql or "{offset:" not in sql:
            raise exceptions.DataError("Scroll queries for ClickHouse need {limit:UInt32} and {offset:UInt32} parameters in the SQL.")
        pages = params.get("pages", 10)
        per_page = params.get("results-per-page", 1000)
        hits = pages_read = 0
        request_context_holder.on_client_request_start()
        request_context_holder.on_request_start()
        try:
            client = await clickhouse.native()
            for page in range(pages):
                page_parameters = {**(parameters or {}), "limit": per_page, "offset": page * per_page}
                result = await client.query(sql, parameters=page_parameters, settings=settings)
                pages_read += 1
                hits += result.row_count
                if result.row_count < per_page:
                    break
        except request_errors() as e:
            self.logger.warning("ClickHouse scroll query failed: %s", e)
            return {**failure(1, "ops", e), "pages": pages_read}
        finally:
            request_context_holder.on_request_end()
            request_context_holder.on_client_request_end()
        return {"weight": 1, "unit": "ops", "success": True, "pages": pages_read, "hits": hits, "hits_relation": "eq",
                "timed_out": False}

    def __repr__(self, *args, **kwargs):
        return "clickhouse-scroll-query"


class ClickHouseVectorSearch(Runner):
    """
    Runs a k-NN SQL query. The query returns one row per neighbor with an ID column (``id-field``, default ``id``).
    ``hnsw_ef_search`` sets ``hnsw_candidate_list_size_for_search``. With ``calculate-recall``, recall is computed
    against ``neighbors``.
    """

    async def __call__(self, clickhouse, params):
        sql, parameters, settings = sql_body(params, self)
        k = params.get("k", 10)
        settings = dict(settings or {})
        if params.get("hnsw_ef_search") is not None:
            settings["hnsw_candidate_list_size_for_search"] = int(params["hnsw_ef_search"])
        request_context_holder.on_client_request_start()
        try:
            client = await clickhouse.native()
            result = await timed(client.query(sql, parameters=parameters, settings=settings or None))
        except request_errors() as e:
            self.logger.warning("ClickHouse vector search failed: %s", e)
            return failure(1, "ops", e)
        finally:
            request_context_holder.on_client_request_end()
        response = {"weight": 1, "unit": "ops", "success": True, "hits": result.row_count, "hits_relation": "eq",
                    "timed_out": False, "took": elapsed_ms(result.summary)}
        if params.get("calculate-recall") and params.get("neighbors") is not None:
            id_field = params.get("id-field", "id")
            if id_field not in result.column_names:
                raise exceptions.DataError(f"The vector search query must return the ID column [{id_field}] to calculate recall.")
            position = result.column_names.index(id_field)
            ids = [row[position] for row in result.result_rows]
            response["recall@k"], response["recall@1"] = recall(ids, to_python(params["neighbors"]), k)
        return response

    def __repr__(self, *args, **kwargs):
        return "clickhouse-vector-search"


def recall(ids, neighbors, k):
    """:return: recall@k and recall@1 of the returned IDs. IDs are compared as strings, so 1 and "1" match."""
    truth = [str(n) for n in neighbors[:k]]
    if not truth:
        return 0.0, 0.0
    returned = [str(i) for i in ids]
    recall_k = len(set(returned[:k]) & set(truth)) / len(truth)
    recall_1 = 1.0 if returned and returned[0] == truth[0] else 0.0
    return recall_k, recall_1


class ClickHouseBulkVectorDataSet(Runner):
    """Inserts a batch of a vector data set. Uses the same insert formats as the bulk runner."""

    def __init__(self):
        super().__init__()
        self._bulk = ClickHouseBulk()

    async def __call__(self, clickhouse, params):
        return await self._bulk(clickhouse, {**params, "bulk-size": params.get("size", 0), "unit": "docs"})

    def __repr__(self, *args, **kwargs):
        return "clickhouse-bulk-vector-data-set"


class ClickHouseCreateTable(Runner):
    """Creates tables with the ``ddl`` (a ``CREATE TABLE`` statement) in the body of each index of the workload."""

    async def __call__(self, clickhouse, params):
        indices = params.get("indices") or [(params.get("index"), params.get("body"))]
        request_context_holder.on_client_request_start()
        request_context_holder.on_request_start()
        try:
            client = await clickhouse.native()
            for name, body in indices:
                ddl = (body or {}).get("ddl")
                if not ddl:
                    raise exceptions.DataError(f"The body of ClickHouse table [{name}] needs a 'ddl' key with the CREATE TABLE statement.")
                await client.command(ddl)
        finally:
            request_context_holder.on_request_end()
            request_context_holder.on_client_request_end()
        return {"weight": len(indices), "unit": "ops", "success": True}

    def __repr__(self, *args, **kwargs):
        return "clickhouse-create-table"


class ClickHouseDropTable(Runner):
    """Drops tables. With ``only-if-exists`` (the default), tables that don't exist are skipped."""

    async def __call__(self, clickhouse, params):
        tables = table_names(params)
        if_exists = " IF EXISTS" if params.get("only-if-exists", True) else ""
        request_context_holder.on_client_request_start()
        request_context_holder.on_request_start()
        try:
            client = await clickhouse.native()
            for table in tables:
                await client.command(f"DROP TABLE{if_exists} {quote_identifier(table)}")
        finally:
            request_context_holder.on_request_end()
            request_context_holder.on_client_request_end()
        return {"weight": len(tables), "unit": "ops", "success": True}

    def __repr__(self, *args, **kwargs):
        return "clickhouse-drop-table"


class ClickHouseFlush(Runner):
    """
    Makes inserted data visible to queries. Inserts are visible once they complete, except asynchronous inserts
    (``async_insert``), which this flushes.
    """

    async def __call__(self, clickhouse, params):
        request_context_holder.on_client_request_start()
        request_context_holder.on_request_start()
        try:
            client = await clickhouse.native()
            await client.command("SYSTEM FLUSH ASYNC INSERT QUEUE")
        finally:
            request_context_holder.on_request_end()
            request_context_holder.on_client_request_end()
        return {"weight": 1, "unit": "ops", "success": True}

    def __repr__(self, *args, **kwargs):
        return "clickhouse-flush"


class ClickHouseOptimizeTable(Runner):
    """Merges all parts of the tables with ``OPTIMIZE TABLE ... FINAL``."""

    async def __call__(self, clickhouse, params):
        tables = table_names(params)
        request_context_holder.on_client_request_start()
        request_context_holder.on_request_start()
        try:
            client = await clickhouse.native()
            tables = await resolve_tables(client, tables)
            for table in tables:
                await client.command(f"OPTIMIZE TABLE {quote_identifier(table)} FINAL")
        finally:
            request_context_holder.on_request_end()
            request_context_holder.on_client_request_end()
        return {"weight": len(tables), "unit": "ops", "success": True}

    def __repr__(self, *args, **kwargs):
        return "clickhouse-optimize-table"


TABLE_STATS = """
SELECT
    (SELECT count() FROM system.parts WHERE active AND database = currentDatabase() AND has({tables:Array(String)}, table)) AS parts,
    (SELECT ifNull(sum(rows), 0) FROM system.parts WHERE active AND database = currentDatabase()
        AND has({tables:Array(String)}, table)) AS rows,
    (SELECT ifNull(sum(bytes_on_disk), 0) FROM system.parts WHERE active AND database = currentDatabase()
        AND has({tables:Array(String)}, table)) AS bytes_on_disk,
    (SELECT count() FROM system.merges WHERE database = currentDatabase() AND has({tables:Array(String)}, table)) AS merges
"""


class ClickHouseTableStats(Runner):
    """
    Reads the number of active parts, rows, bytes on disk and running merges of the tables. With a ``condition``
    (``{"path": "merges", "expected-value": 0}``), the operation succeeds only if the statistic has the expected
    value, so it can wait with ``retry-until-success``.
    """

    async def __call__(self, clickhouse, params):
        tables = table_names(params)
        request_context_holder.on_client_request_start()
        request_context_holder.on_request_start()
        try:
            client = await clickhouse.native()
            tables = await resolve_tables(client, tables)
            result = await client.query(TABLE_STATS, parameters={"tables": tables})
        finally:
            request_context_holder.on_request_end()
            request_context_holder.on_client_request_end()
        stats = dict(zip(result.column_names, (int(v) for v in result.result_rows[0])))
        response = {"weight": 1, "unit": "ops", "success": True, "stats": stats}
        condition = params.get("condition")
        if condition:
            path, expected = condition["path"], condition["expected-value"]
            if path not in stats:
                raise exceptions.DataError(f"Unknown ClickHouse table statistic [{path}]. Available: {', '.join(stats)}.")
            response["success"] = stats[path] == expected
            response["condition"] = {"path": path, "actual-value": stats[path], "expected-value": expected}
        return response

    def __repr__(self, *args, **kwargs):
        return "clickhouse-table-stats"


class ClickHousePing(Runner):
    """Checks that ClickHouse answers requests. Use it with ``retry-until-success`` to wait for ClickHouse."""

    async def __call__(self, clickhouse, params):
        request_context_holder.on_client_request_start()
        request_context_holder.on_request_start()
        try:
            client = await clickhouse.native()
            available = await client.ping()
        except request_errors() as e:
            self.logger.debug("ClickHouse ping failed: %s", e)
            available = False
        finally:
            request_context_holder.on_request_end()
            request_context_holder.on_client_request_end()
        return {"weight": 1, "unit": "ops", "success": bool(available)}

    def __repr__(self, *args, **kwargs):
        return "clickhouse-ping"


def runners():
    """:return: The ClickHouse runners keyed by operation type."""
    return {
        workload.OperationType.Bulk: ClickHouseBulk(),
        workload.OperationType.BulkVectorDataSet: ClickHouseBulkVectorDataSet(),
        workload.OperationType.Search: ClickHouseQuery(),
        workload.OperationType.PaginatedSearch: ClickHouseQuery(),
        workload.OperationType.ScrollSearch: ClickHouseScrollQuery(),
        workload.OperationType.VectorSearch: ClickHouseVectorSearch(),
        workload.OperationType.CreateIndex: ClickHouseCreateTable(),
        workload.OperationType.DeleteIndex: ClickHouseDropTable(),
        workload.OperationType.Refresh: ClickHouseFlush(),
        workload.OperationType.ForceMerge: ClickHouseOptimizeTable(),
        workload.OperationType.IndexStats: Retry(ClickHouseTableStats()),
        workload.OperationType.ClusterHealth: Retry(ClickHousePing()),
    }
