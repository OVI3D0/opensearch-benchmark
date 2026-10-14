# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""
The contract every database engine implements.

An engine tells OSB how to talk to one kind of database: how to create its
clients, how to tell whether it is reachable, which runners execute workload
operations against it, which operations it supports, and which telemetry
devices apply to it. The worker coordinator and test execution orchestrator
only call these methods, so they never need to know which engine is selected.
"""

import logging
from abc import ABC, abstractmethod

from osbenchmark import config, exceptions, workload

#: Operation types every engine can run because they never send a request to the database.
ENGINE_AGNOSTIC_OPERATIONS = frozenset([workload.OperationType.Sleep])


class Engine(ABC):
    """
    Base class for database engines.

    Subclasses must set ``name`` and implement the abstract client and readiness methods.
    The remaining methods have defaults suitable for an engine with no
    engine-specific runners or telemetry.
    """

    #: The value users pass to ``--database-type``.
    name = None

    #: The engine's name as shown in messages.
    display_name = None

    #: Distribution version used to pick the workload branch when the engine
    #: cannot report a version of its own.
    default_distribution_version = "3.0.0"

    #: Whether OSB's built-in runners, which are written against the OpenSearch API, work with this engine.
    #: Engines that do not use them can only run the operations they provide runners for.
    uses_default_runners = False

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def create_client(self, hosts, client_options):
        """
        Create a synchronous client. Used before the benchmark starts and by telemetry.

        :param hosts: The hosts of a single cluster.
        :param client_options: Client options for that cluster.
        """

    @abstractmethod
    def create_async_client(self, hosts, client_options, cfg):
        """
        Create an asynchronous client. Used by workers to run workload operations.

        :param hosts: The hosts of a single cluster.
        :param client_options: Client options for that cluster.
        :param cfg: The current config object.
        """

    @abstractmethod
    async def close_async_client(self, client):
        """
        Close a client created by ``create_async_client`` once all workload operations have finished.

        :param client: The client to close.
        """

    @abstractmethod
    def wait_for_rest_layer(self, client, max_attempts=40):
        """
        Wait until the engine accepts requests.

        :param client: A client created by ``create_client``.
        :param max_attempts: How many times to probe before giving up.
        :return: ``True`` if the engine became available, ``False`` otherwise.
        """

    def cluster_info(self, client):
        """
        :param client: A client created by ``create_client``.
        :return: Cluster details reported at benchmark start, or ``None`` if the engine has none.
        """
        return None

    def configure_distribution_version(self, cfg, sources=False):
        """
        Make sure ``builder.distribution.version`` is set before the workload is loaded. It selects the
        workload branch and is available to workload templates.

        :param cfg: The current config object.
        :param sources: Whether OSB builds the engine from sources.
        """
        if not cfg.exists("builder", "distribution.version"):
            self.logger.info("Using default distribution version [%s] for database type [%s].",
                             self.default_distribution_version, self.name)
            cfg.add(config.Scope.benchmark, "builder", "distribution.version", self.default_distribution_version)

    def runners(self):
        """
        :return: Engine-specific runners keyed by operation type. OSB registers them after the default runners and
                 workload plugin runners, so they take precedence.
        """
        return {}

    def unsupported_operations(self):
        """
        :return: Built-in operation types this engine cannot run, although it uses the default runners.
        """
        return set()

    def supports_operation(self, operation_type):
        """
        :param operation_type: An operation type as written in a workload, e.g. ``bulk``.
        :return: Whether this engine can run operations of that type. Operation types that are not built in,
                 such as ones provided by workload plugins, are always considered supported.
        """
        try:
            operation = workload.OperationType.from_hyphenated_string(operation_type)
        except KeyError:
            return True
        if operation in ENGINE_AGNOSTIC_OPERATIONS:
            return True
        if self.uses_default_runners:
            return operation not in self.unsupported_operations()
        return operation in self._registered_operations()

    def _registered_operations(self):
        registered = set()
        for operation_type in self.runners():
            if isinstance(operation_type, workload.OperationType):
                registered.add(operation_type)
            else:
                try:
                    registered.add(workload.OperationType.from_hyphenated_string(operation_type))
                except KeyError:
                    pass
        return registered

    def check_test_procedure(self, test_procedure):
        """
        :param test_procedure: The test procedure about to run.
        :raises SystemSetupError: If the test procedure contains operations this engine cannot run.
        """
        unsupported = [f"[{task.name}] ({task.operation.type})"
                       for step in test_procedure.schedule
                       for task in step
                       if not self.supports_operation(task.operation.type)]
        if unsupported:
            raise exceptions.SystemSetupError(
                f"Database type [{self.name}] does not support these tasks in test procedure [{test_procedure.name}]: "
                f"{', '.join(unsupported)}. Remove them with --exclude-tasks, or use a workload written for "
                f"{self.display_name}.")

    def telemetry_devices(self, cfg, clients, metrics_store, workload):
        """
        :param cfg: The current config object.
        :param clients: Synchronous clients keyed by cluster name.
        :param metrics_store: The metrics store telemetry devices write to.
        :param workload: The workload being run.
        :return: The telemetry devices this engine supports.
        """
        return []
