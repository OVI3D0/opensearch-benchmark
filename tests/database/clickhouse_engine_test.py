# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import asyncio
from unittest import mock

import pytest

from osbenchmark import exceptions
from osbenchmark.database import get_engine
from osbenchmark.database.clients.clickhouse import engine as clickhouse_engine
from osbenchmark.database.clients.clickhouse import helpers

HOSTS = [{"host": "127.0.0.1", "port": 8123}]


def test_connection_args_map_osb_client_options():
    args = clickhouse_engine.connection_args(HOSTS, {"basic_auth_user": "u", "basic_auth_password": "p", "timeout": 60,
                                                     "verify_certs": False, "clickhouse_settings": {"max_threads": 4}})

    assert args["host"] == "127.0.0.1"
    assert args["port"] == 8123
    assert args["secure"] is False
    assert (args["username"], args["password"]) == ("u", "p")
    assert args["send_receive_timeout"] == 60
    assert args["verify"] is False
    assert args["settings"] == {"max_threads": 4}
    assert args["query_retries"] == 0


def test_https_port_is_secure_and_native_port_is_rejected():
    assert helpers.parse_hosts([{"host": "h", "port": 8443}]) == ("h", 8443, True)
    with pytest.raises(exceptions.SystemSetupError, match="native protocol"):
        helpers.parse_hosts([{"host": "h", "port": 9000}])


@mock.patch.object(clickhouse_engine, "clickhouse_connect")
def test_async_client_pool_is_sized_by_max_connections(module):
    module.return_value.get_async_client = mock.AsyncMock(return_value=mock.Mock(close=mock.AsyncMock()))
    engine = get_engine("clickhouse")

    client = engine.create_async_client(HOSTS, {"max_connections": 256}, cfg=None)
    native = asyncio.run(client.native())
    asyncio.run(engine.close_async_client(client))

    kwargs = module.return_value.get_async_client.await_args.kwargs
    assert (kwargs["connector_limit"], kwargs["connector_limit_per_host"]) == (256, 256)
    native.close.assert_awaited_once_with()


@mock.patch.object(clickhouse_engine, "clickhouse_connect")
def test_sync_client_connects_on_first_use(module):
    engine = get_engine("clickhouse")

    client = engine.create_client(HOSTS, {})
    module.return_value.get_client.assert_not_called()
    module.return_value.get_client.return_value.server_version = "26.8.34.13"

    assert engine.cluster_info(client)["version"]["number"] == "26.8.34"
    module.return_value.get_client.assert_called_once()


@mock.patch("time.sleep")
def test_waits_until_clickhouse_answers(_):
    client = mock.Mock()
    client.native.return_value.ping.side_effect = [False, ConnectionError("refused"), True]

    assert get_engine("clickhouse").wait_for_rest_layer(client, max_attempts=5) is True


def test_missing_client_library_explains_how_to_install():
    with mock.patch.dict("sys.modules", {"clickhouse_connect": None}):
        with pytest.raises(exceptions.SystemSetupError, match=r"opensearch-benchmark\[clickhouse\]"):
            clickhouse_engine.clickhouse_connect()


def test_clickhouse_supports_only_operations_with_runners():
    engine = get_engine("clickhouse")

    assert engine.supports_operation("search")
    assert engine.supports_operation("bulk")
    assert not engine.supports_operation("put-pipeline")
    assert not engine.supports_operation("create-index-template")


def test_helpers():
    assert helpers.quote_identifier("a`b") == "`a``b`"
    assert helpers.parse_version("26.8.34.13-lts") == "26.8.34"
    assert helpers.elapsed_ms({"elapsed_ns": "1500000"}) == 1.5
    assert helpers.elapsed_ms(None) == 0
