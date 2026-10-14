# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import pytest

from osbenchmark import exceptions, workload
from osbenchmark.database.clients.opensearch import OpenSearchEngine
from osbenchmark.database.engine import Engine


class VectorEngine(Engine):
    """An engine with its own runners for bulk indexing and vector search only."""
    name = "vector"
    display_name = "Vector"

    def create_client(self, hosts, client_options):
        return None

    def create_async_client(self, hosts, client_options, cfg):
        return None

    async def close_async_client(self, client):
        pass

    def wait_for_rest_layer(self, client, max_attempts=40):
        return True

    def runners(self):
        return {workload.OperationType.Bulk: object(), "vector-search": object()}


class DefaultRunnerEngine(VectorEngine):
    """An engine that uses the default runners except for ML operations."""
    name = "default-runner"
    uses_default_runners = True

    def unsupported_operations(self):
        return {workload.OperationType.RegisterMlModel}


def task(name, operation_type):
    return workload.Task(name, workload.Operation(name, operation_type))


def test_procedure(*schedule):
    return workload.TestProcedure("default", schedule=list(schedule))


@pytest.mark.parametrize("operation_type", ["bulk", "vector-search"])
def test_engine_supports_operations_it_provides_runners_for(operation_type):
    assert VectorEngine().supports_operation(operation_type)


def test_engine_does_not_support_other_built_in_operations():
    assert not VectorEngine().supports_operation("create-index")
    assert not VectorEngine().supports_operation("put-pipeline")


def test_every_engine_supports_engine_agnostic_operations():
    assert VectorEngine().supports_operation("sleep")


def test_operations_that_are_not_built_in_are_supported():
    # e.g. provided by a workload plugin
    assert VectorEngine().supports_operation("my-custom-operation")


def test_default_runner_engine_supports_built_in_operations_except_excluded_ones():
    engine = DefaultRunnerEngine()

    assert engine.supports_operation("create-index")
    assert not engine.supports_operation("register-ml-model")


def test_opensearch_supports_every_built_in_operation():
    engine = OpenSearchEngine()

    assert all(engine.supports_operation(operation.to_hyphenated_string()) for operation in workload.OperationType)


def test_supported_test_procedure_passes():
    VectorEngine().check_test_procedure(test_procedure(
        task("index", "bulk"),
        workload.Parallel([task("query", "vector-search"), task("pause", "sleep")]),
    ))


def test_unsupported_tasks_are_reported_including_parallel_ones():
    procedure = test_procedure(
        task("create", "create-index"),
        task("index", "bulk"),
        workload.Parallel([task("query", "vector-search"), task("pipeline", "put-pipeline")]),
    )

    with pytest.raises(exceptions.SystemSetupError) as error:
        VectorEngine().check_test_procedure(procedure)

    assert error.value.args[0] == (
        "Database type [vector] does not support these tasks in test procedure [default]: "
        "[create] (create-index), [pipeline] (put-pipeline). Remove them with --exclude-tasks, "
        "or use a workload written for Vector.")
