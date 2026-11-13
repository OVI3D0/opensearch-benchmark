# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import asyncio
import time

from osbenchmark import exceptions
from osbenchmark.context import RequestContextHolder
from osbenchmark.database.clients.clickhouse.helpers import parse_hosts, parse_version
from osbenchmark.database.engine import Engine

# Concurrent connections per worker when OSB does not size the pool from the number of clients.
DEFAULT_MAX_CONNECTIONS = 100


def clickhouse_connect():
    try:
        import clickhouse_connect as module  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise exceptions.SystemSetupError(
            "The ClickHouse client is not installed. Install it with: pip install 'opensearch-benchmark[clickhouse]'") from e
    return module


def connection_args(hosts, client_options):
    """
    Builds clickhouse-connect connection arguments from OSB's hosts and client options.

    Supported client options: ``basic_auth_user`` / ``basic_auth_password`` (or ``username`` / ``password``),
    ``database``, ``use_ssl``, ``verify_certs``, ``ca_certs``, ``timeout``, ``connect_timeout``, ``compress`` and
    ``clickhouse_settings`` (ClickHouse settings applied to every request).
    """
    host, port, secure = parse_hosts(hosts)
    options = client_options or {}
    args = {
        "host": host,
        "port": port,
        "secure": bool(options.get("use_ssl", secure)),
        "username": options.get("basic_auth_user", options.get("username", "default")),
        "password": options.get("basic_auth_password", options.get("password", "")),
        "database": options.get("database", "default"),
        "verify": options.get("verify_certs", True),
        "connect_timeout": options.get("connect_timeout", 10),
        "send_receive_timeout": options.get("timeout", 300),
        "compress": options.get("compress", True),
        # A benchmark must see every failure, so don't let the client retry silently.
        "query_retries": 0,
    }
    if options.get("ca_certs"):
        args["ca_cert"] = options["ca_certs"]
    if options.get("clickhouse_settings"):
        args["settings"] = dict(options["clickhouse_settings"])
    return args


class ClickHouseClient(RequestContextHolder):
    """
    Holds the native clickhouse-connect async client of a worker. The native client can only be created inside the
    event loop, so it is created on first use and shared by all clients of the worker.
    """

    def __init__(self, connect):
        """
        :param connect: A coroutine function that creates the native async client.
        """
        self._connect = connect
        self._client = None
        self._lock = None

    async def native(self):
        """:return: The native ``clickhouse_connect`` async client."""
        if self._client is None:
            if self._lock is None:
                self._lock = asyncio.Lock()
            async with self._lock:
                if self._client is None:
                    self._client = await self._connect()
        return self._client

    async def close(self):
        if self._client is not None:
            await self._client.close()
            self._client = None


class ClickHouseSyncClient:
    """
    Holds a native clickhouse-connect sync client. Creating the native client already connects to ClickHouse, so it is
    created on first use, which lets OSB wait for ClickHouse to become available first.
    """

    def __init__(self, connect):
        """
        :param connect: A function that creates the native sync client.
        """
        self._connect = connect
        self._client = None

    def native(self):
        """:return: The native ``clickhouse_connect`` sync client."""
        if self._client is None:
            self._client = self._connect()
        return self._client


class ClickHouseEngine(Engine):
    """ClickHouse, accessed over HTTP with clickhouse-connect. Workloads send SQL; see the ClickHouse runners."""

    name = "clickhouse"
    display_name = "ClickHouse"

    def create_client(self, hosts, client_options):
        module = clickhouse_connect()
        args = connection_args(hosts, client_options)
        return ClickHouseSyncClient(lambda: module.get_client(**args))

    def create_async_client(self, hosts, client_options, cfg):
        module = clickhouse_connect()
        args = connection_args(hosts, client_options)
        # every client of a worker shares one connection pool, sized by OSB for the number of clients
        max_connections = (client_options or {}).get("max_connections", DEFAULT_MAX_CONNECTIONS)
        args["connector_limit"] = max_connections
        args["connector_limit_per_host"] = max_connections

        async def connect():
            return await module.get_async_client(**args)

        return ClickHouseClient(connect)

    async def close_async_client(self, client):
        await client.close()

    def wait_for_rest_layer(self, client, max_attempts=40):
        for attempt in range(max_attempts):
            try:
                if client.native().ping():
                    self.logger.info("ClickHouse is available after [%d] attempt(s).", attempt + 1)
                    return True
            except Exception as e:  # pylint: disable=broad-except
                self.logger.debug("ClickHouse is not available yet: %s", e)
            time.sleep(3)
        return False

    def cluster_info(self, client):
        version = client.native().server_version
        return {
            "name": "clickhouse",
            "version": {
                "distribution": "clickhouse",
                "number": parse_version(version) or version,
                "build_hash": None,
            }
        }

    def runners(self):
        # imported here because the runners depend on the worker coordinator
        from osbenchmark.worker_coordinator.runners import clickhouse  # pylint: disable=import-outside-toplevel
        return clickhouse.runners()
