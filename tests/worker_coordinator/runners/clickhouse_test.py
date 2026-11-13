# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import asyncio
import json
from types import SimpleNamespace
from unittest import mock

import pytest
from clickhouse_connect.driver.exceptions import DatabaseError

from osbenchmark import exceptions, workload
from osbenchmark.context import RequestContextHolder
from osbenchmark.worker_coordinator.runner import Retry
from osbenchmark.worker_coordinator.runners import clickhouse as runners


def result(rows=(), columns=(), summary=None):
    return SimpleNamespace(result_rows=list(rows), column_names=tuple(columns), row_count=len(rows),
                           summary=summary or {"elapsed_ns": "2500000"})


class FakeClickHouse:
    """Stands in for ClickHouseClient: native() returns a mock of the clickhouse-connect async client."""

    def __init__(self):
        self.client = mock.Mock()
        for method in ("query", "insert", "raw_insert", "command", "ping"):
            setattr(self.client, method, mock.AsyncMock())
        for method in ("insert", "raw_insert"):
            getattr(self.client, method).return_value = SimpleNamespace(summary={"elapsed_ns": "2500000"})

    async def native(self):
        return self.client


def run(runner, params, clickhouse=None):
    clickhouse = clickhouse or FakeClickHouse()

    async def run_in_request_context():
        async with RequestContextHolder().new_request_context():
            return await runner(clickhouse, params)

    return asyncio.run(run_in_request_context()), clickhouse.client


BULK_NDJSON = '{"index": {"_index": "logs"}}\n{"id": 1, "path": "/a"}\n{"create": {}}\n{"id": 2, "path": "/b"}\n'


def test_bulk_passes_ndjson_documents_through_as_json_each_row():
    response, client = run(runners.ClickHouseBulk(), {"index": "logs", "body": BULK_NDJSON, "bulk-size": 2})

    assert response == {"weight": 2, "unit": "docs", "success": True, "error-count": 0, "took": 2.5}
    args = client.raw_insert.await_args
    assert args.args == ("logs",)
    assert args.kwargs["fmt"] == "JSONEachRow"
    assert args.kwargs["insert_block"] == b'{"id": 1, "path": "/a"}\n{"id": 2, "path": "/b"}\n'


def test_bulk_with_column_names_inserts_rows():
    body = [{"index": {}}, {"id": 1, "path": "/a", "extra": "x"}, {"index": {}}, {"id": 2, "path": "/b"}]
    response, client = run(runners.ClickHouseBulk(), {"index": "logs", "body": body, "bulk-size": 2, "column-names": ["id", "path"]})

    assert response["success"] is True
    client.insert.assert_awaited_once_with("logs", [(1, "/a"), (2, "/b")], column_names=["id", "path"], settings=None)


def test_bulk_rejects_update_actions():
    with pytest.raises(exceptions.BenchmarkError, match="only 'index' and 'create'"):
        run(runners.ClickHouseBulk(), {"index": "logs", "body": '{"update": {}}\n{"id": 1}\n', "bulk-size": 1})


def test_failed_bulk_counts_as_one_error():
    clickhouse = FakeClickHouse()
    clickhouse.client.raw_insert.side_effect = DatabaseError("Code: 27. Cannot parse input")

    response, _ = run(runners.ClickHouseBulk(), {"index": "logs", "body": BULK_NDJSON, "bulk-size": 2}, clickhouse)

    assert response["success"] is False
    assert response["error-count"] == 1
    assert response["weight"] == 2
    assert "Cannot parse input" in response["error-description"]


def test_query_reports_rows_as_hits():
    clickhouse = FakeClickHouse()
    clickhouse.client.query.return_value = result(rows=[("/a", 3), ("/b", 4)], columns=("path", "c"))
    body = {"sql": "SELECT path, count() c FROM logs WHERE status = {s:UInt16} GROUP BY path", "parameters": {"s": 500}}

    response, client = run(runners.ClickHouseQuery(), {"body": body}, clickhouse)

    assert response == {"weight": 1, "unit": "ops", "success": True, "hits": 2, "hits_relation": "eq", "timed_out": False, "took": 2.5}
    client.query.assert_awaited_once_with(body["sql"], parameters={"s": 500}, settings=None)


def test_query_without_sql_explains_that_clickhouse_needs_sql():
    with pytest.raises(exceptions.DataError, match="ClickHouse workloads send SQL"):
        run(runners.ClickHouseQuery(), {"body": {"query": {"match_all": {}}}})


def test_failed_query_is_reported_as_error():
    clickhouse = FakeClickHouse()
    clickhouse.client.query.side_effect = DatabaseError("Code: 47. Unknown identifier")

    response, _ = run(runners.ClickHouseQuery(), {"body": {"sql": "SELECT nope"}}, clickhouse)

    assert response["success"] is False
    assert response["error-type"] == "clickhouse"


def test_scroll_stops_at_last_page():
    clickhouse = FakeClickHouse()
    clickhouse.client.query.side_effect = [result(rows=[(1,)] * 2), result(rows=[(1,)])]
    sql = "SELECT id FROM logs ORDER BY id LIMIT {limit:UInt32} OFFSET {offset:UInt32}"

    response, client = run(runners.ClickHouseScrollQuery(), {"body": {"sql": sql}, "pages": 5, "results-per-page": 2}, clickhouse)

    assert response["pages"] == 2
    assert response["hits"] == 3
    assert client.query.await_args_list[1].kwargs["parameters"] == {"limit": 2, "offset": 2}


def test_scroll_needs_limit_and_offset_parameters():
    with pytest.raises(exceptions.DataError, match="limit:UInt32"):
        run(runners.ClickHouseScrollQuery(), {"body": {"sql": "SELECT id FROM logs"}})


def test_vector_search_calculates_recall():
    clickhouse = FakeClickHouse()
    clickhouse.client.query.return_value = result(rows=[(7, 0.1), (3, 0.2), (9, 0.3)], columns=("id", "distance"))
    params = {"body": {"sql": "SELECT id, distance FROM v"}, "k": 3, "calculate-recall": True, "neighbors": ["7", "1", "9"],
              "hnsw_ef_search": 64}

    response, client = run(runners.ClickHouseVectorSearch(), params, clickhouse)

    assert response["recall@k"] == pytest.approx(2 / 3)
    assert response["recall@1"] == 1.0
    assert client.query.await_args.kwargs["settings"] == {"hnsw_candidate_list_size_for_search": 64}


def test_create_table_runs_ddl_of_each_index():
    indices = [("a", {"ddl": "CREATE TABLE a (x UInt8) ENGINE = MergeTree ORDER BY x"}), ("b", {"ddl": "CREATE TABLE b (y UInt8) ENGINE = Log"})]

    response, client = run(runners.ClickHouseCreateTable(), {"indices": indices})

    assert response["weight"] == 2
    assert [c.args[0] for c in client.command.await_args_list] == [indices[0][1]["ddl"], indices[1][1]["ddl"]]


def test_create_table_needs_ddl():
    with pytest.raises(exceptions.DataError, match="'ddl' key"):
        run(runners.ClickHouseCreateTable(), {"indices": [("a", {"settings": {}})]})


def test_drop_table_quotes_names():
    _, client = run(runners.ClickHouseDropTable(), {"indices": ["logs", "db.other"], "only-if-exists": True})

    assert [c.args[0] for c in client.command.await_args_list] == ["DROP TABLE IF EXISTS `logs`", "DROP TABLE IF EXISTS `db`.`other`"]


def test_optimize_resolves_all_tables():
    clickhouse = FakeClickHouse()
    clickhouse.client.query.return_value = result(rows=[("a",), ("b",)])

    response, client = run(runners.ClickHouseOptimizeTable(), {"index": "_all"}, clickhouse)

    assert response["weight"] == 2
    assert [c.args[0] for c in client.command.await_args_list] == ["OPTIMIZE TABLE `a` FINAL", "OPTIMIZE TABLE `b` FINAL"]


def test_table_stats_condition():
    clickhouse = FakeClickHouse()
    clickhouse.client.query.return_value = result(rows=[(3, 100, 2048, 1)], columns=("parts", "rows", "bytes_on_disk", "merges"))

    response, client = run(runners.ClickHouseTableStats(), {"index": "logs", "condition": {"path": "merges", "expected-value": 0}},
                           clickhouse)

    assert response["stats"] == {"parts": 3, "rows": 100, "bytes_on_disk": 2048, "merges": 1}
    assert response["success"] is False
    assert client.query.await_args.kwargs["parameters"] == {"tables": ["logs"]}


def test_table_stats_rejects_unknown_statistic():
    clickhouse = FakeClickHouse()
    clickhouse.client.query.return_value = result(rows=[(1, 1, 1, 0)], columns=("parts", "rows", "bytes_on_disk", "merges"))

    with pytest.raises(exceptions.DataError, match="Unknown ClickHouse table statistic"):
        run(runners.ClickHouseTableStats(), {"index": "logs", "condition": {"path": "_all.total.merges.current", "expected-value": 0}},
            clickhouse)


def test_ping_failure_is_unsuccessful_not_an_error():
    clickhouse = FakeClickHouse()
    clickhouse.client.ping.side_effect = DatabaseError("connection refused")

    response, _ = run(runners.ClickHousePing(), {}, clickhouse)

    assert response == {"weight": 1, "unit": "ops", "success": False}


def test_runners_cover_supported_operations_and_retry_waiting_operations():
    registered = runners.runners()

    assert workload.OperationType.PutPipeline not in registered
    assert isinstance(registered[workload.OperationType.ClusterHealth], Retry)
    assert isinstance(registered[workload.OperationType.IndexStats], Retry)


def test_bulk_body_from_bytes_with_unicode():
    body = ('{"index": {}}\n' + json.dumps({"path": "/ü"}) + "\n").encode("utf-8")

    _, client = run(runners.ClickHouseBulk(), {"index": "logs", "body": body, "bulk-size": 1})

    assert client.raw_insert.await_args.kwargs["insert_block"] == (json.dumps({"path": "/ü"}) + "\n").encode("utf-8")
