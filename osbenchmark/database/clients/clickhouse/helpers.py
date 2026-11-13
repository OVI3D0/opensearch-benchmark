# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""
Helpers for the ClickHouse engine that don't talk to ClickHouse.
"""

import json
import logging
import re

from osbenchmark import exceptions

logger = logging.getLogger(__name__)

SUPPORTED_BULK_ACTIONS = frozenset(["index", "create"])
BULK_ACTIONS = ("index", "create", "update", "delete")

# clickhouse-connect talks HTTP. These ports default to TLS.
SECURE_PORTS = frozenset([8443, 9440])
NATIVE_PROTOCOL_PORT = 9000


def parse_hosts(hosts):
    """
    :param hosts: The hosts of one cluster, as a list of {"host": ..., "port": ...} dicts.
    :return: A tuple (host, port, secure) for the first host. ClickHouse connections go to a single endpoint.
    """
    if not hosts:
        raise exceptions.SystemSetupError("No ClickHouse hosts are configured.")
    if len(hosts) > 1:
        logger.warning("ClickHouse uses only the first of [%d] configured hosts: [%s].", len(hosts), hosts[0])
    first = hosts[0]
    host = first.get("host", "localhost")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    try:
        port = int(first.get("port", 8123))
    except (TypeError, ValueError) as e:
        raise exceptions.SystemSetupError(f"Invalid ClickHouse port [{first.get('port')}].") from e
    if port == NATIVE_PROTOCOL_PORT:
        raise exceptions.SystemSetupError(
            f"Port [{NATIVE_PROTOCOL_PORT}] is ClickHouse's native protocol port. OSB connects over HTTP: use port 8123, or 8443 "
            f"for HTTPS.")
    return host, port, bool(first.get("use_ssl", port in SECURE_PORTS))


def quote_identifier(name):
    """
    :param name: A table or database name from a workload. Use query parameters for values.
    :return: The name as a quoted ClickHouse identifier. ``db.table`` names are quoted part by part.
    """
    if not isinstance(name, str) or not name:
        raise exceptions.BenchmarkError(f"Invalid ClickHouse identifier [{name!r}].")
    if any(c in name for c in "\x00\r\n"):
        raise exceptions.BenchmarkError(f"ClickHouse identifier [{name!r}] contains a control character.")
    return ".".join("`" + part.replace("`", "``") + "`" for part in name.split(".", 1))


def parse_bulk_body(body):
    """
    Turns an OSB bulk body into the documents to insert. Only ``index`` and ``create`` actions are supported, since
    ClickHouse inserts are append-only.

    :param body: Newline-delimited JSON (str or bytes), or a list that alternates action metadata and documents.
    :return: A list of documents (the ``_source`` of each action).
    """
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")
    if isinstance(body, str):
        lines = []
        for line_number, line in enumerate(line for line in body.split("\n") if line.strip()):
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise exceptions.BenchmarkError(f"Line [{line_number}] of the bulk body is not valid JSON: {e}") from e
        body = lines
    if not isinstance(body, list):
        raise exceptions.BenchmarkError(f"Unsupported bulk body of type [{type(body).__name__}].")
    if len(body) % 2 != 0:
        raise exceptions.BenchmarkError("The bulk body has an action without a document.")
    docs = []
    for i in range(0, len(body), 2):
        action = next((key for key in BULK_ACTIONS if key in body[i]), None)
        if action not in SUPPORTED_BULK_ACTIONS:
            raise exceptions.BenchmarkError(
                f"ClickHouse supports only 'index' and 'create' bulk actions but the bulk body contains [{action or body[i]}].")
        docs.append(body[i + 1])
    return docs


def to_python(value):
    """Converts numpy arrays and scalars, e.g. from vector data sets, to Python values that clickhouse-connect accepts."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_python(v) for v in value)
    return value


def elapsed_ms(summary):
    """
    :param summary: The ``X-ClickHouse-Summary`` of a response, as returned by clickhouse-connect.
    :return: The server-side elapsed time in milliseconds, or 0 if it is not available.
    """
    try:
        return int((summary or {}).get("elapsed_ns", 0)) / 1_000_000
    except (TypeError, ValueError):
        return 0


SEMVER = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def parse_version(version):
    """
    :param version: A ClickHouse version string, e.g. ``25.8.1.5101-stable``.
    :return: The ``major.minor.patch`` part, or None if it can't be parsed.
    """
    m = SEMVER.match(version or "")
    return ".".join(m.groups()) if m else None
