# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""
Base runner classes and utilities for OpenSearch Benchmark.

This module contains the core runner infrastructure that is shared
across all database-specific runner implementations. The classes here
are database-agnostic and provide the foundation for building
runners for any database backend.

Key components:
- Runner: Base class for all benchmark operations
- Delegator: Mixin for delegation patterns (retry, assertions, etc.)
- time_func: Decorator for request timing
- Utility functions: mandatory(), escape(), remove_prefix()
"""

import logging
import types

from osbenchmark import exceptions, workload
from osbenchmark.context import RequestContextHolder

# Shared request context holder - used for timing across all runners
request_context_holder = RequestContextHolder()


def mandatory(params, key, op):
    """
    Get a mandatory parameter from the params dict.

    Args:
        params: Parameter dictionary
        key: Key to look up
        op: Runner/operation instance (for error messages)

    Returns:
        The parameter value

    Raises:
        DataError: If the key is missing
    """
    try:
        return params[key]
    except KeyError:
        raise exceptions.DataError(
            f"Parameter source for operation '{str(op)}' did not provide the mandatory parameter '{key}'. "
            f"Add it to your parameter source and try again.")


def remove_prefix(string, prefix):
    """
    Remove a prefix from a string if present.

    TODO: Remove and use str.removeprefix() once Python 3.9 is the minimum version.

    Args:
        string: The string to process
        prefix: The prefix to remove

    Returns:
        The string with the prefix removed, or unchanged if prefix not present
    """
    if string.startswith(prefix):
        return string[len(prefix):]
    return string


def escape(v):
    """
    Escape values so they can be used as query parameters.

    Args:
        v: The raw value. May be None.

    Returns:
        The escaped value as a string, or None if input is None.
    """
    if v is None:
        return None
    elif isinstance(v, bool):
        return str(v).lower()
    else:
        return str(v)


def time_func(func):
    """
    Decorator to wrap a function with request timing.

    Calls on_client_request_start() before and on_client_request_end() after
    the function execution. This tracks the time spent in client operations.

    Args:
        func: The async function to wrap

    Returns:
        The wrapped async function with timing
    """
    async def advised(*args, **kwargs):
        request_context_holder.on_client_request_start()
        try:
            response = await func(*args, **kwargs)
            return response
        finally:
            request_context_holder.on_client_request_end()
    return advised


class Runner:
    """
    Base class for all benchmark operations.

    Subclasses must implement the __call__ method to perform the actual
    benchmark operation against the database.

    The __call__ method should return a dict with at least:
    - weight: Number of operations performed (typically 1, or bulk size for bulk ops)
    - unit: Unit of measurement (typically "ops" or "docs")
    - success: Boolean indicating if the operation succeeded

    Additional metrics can be included in the return dict and will be recorded.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        return self

    async def __call__(self, client, params):
        """
        Run the benchmark operation.

        Args:
            client: Database client instance
            params: Operation parameters from the workload

        Returns:
            A dict with operation results and metrics

        Raises:
            NotImplementedError: If the subclass doesn't implement this method
        """
        raise NotImplementedError("abstract operation")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    def _default_kw_params(self, params):
        """
        Extract common API keyword parameters from workload params.

        This maps workload parameter names to API keyword argument names.
        Override in subclasses if different parameter mappings are needed.

        Args:
            params: The workload parameters dict

        Returns:
            A dict of keyword arguments for the API call
        """
        kw_dict = {
            "body": "body",
            "headers": "headers",
            "index": "index",
            "opaque_id": "opaque-id",
            "params": "request-params",
            "request_timeout": "request-timeout",
        }
        full_result = {k: params.get(v) for (k, v) in kw_dict.items()}
        # Filter out None values
        return dict(filter(lambda kv: kv[1] is not None, full_result.items()))

    def _transport_request_params(self, params):
        """
        Extract transport-level request parameters from workload params.

        Args:
            params: The workload parameters dict

        Returns:
            A tuple of (request_params dict, headers dict)
        """
        request_params = params.get("request-params", {})
        request_timeout = params.get("request-timeout")
        if request_timeout is not None:
            request_params["request_timeout"] = request_timeout
        headers = params.get("headers") or {}
        opaque_id = params.get("opaque-id")
        if opaque_id is not None:
            headers.update({"x-opaque-id": opaque_id})
        return request_params, headers


class Delegator:
    """
    Mixin class to enable delegation patterns.

    This is used by wrapper classes (like Retry, AssertingRunner) that
    delegate to an underlying runner while adding behavior.
    """

    def __init__(self, delegate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delegate = delegate


def unwrap(runner):
    """
    Unwrap a delegating runner to get the underlying runner.

    Recursively unwraps delegation chains until the actual runner is found.

    Args:
        runner: An arbitrarily nested chain of delegators around a runner

    Returns:
        The innermost runner instance
    """
    delegate = getattr(runner, "delegate", None)
    if delegate:
        return unwrap(delegate)
    else:
        return runner


def _single_cluster_runner(runnable, name, context_manager_enabled=False):
    """
    Wrap a runner to extract the default cluster client.

    Args:
        runnable: The runner to wrap
        name: Name for the runner (for repr)
        context_manager_enabled: Whether the runner supports context manager

    Returns:
        A MultiClientRunner that extracts the "default" client
    """
    return MultiClientRunner(runnable, name, lambda clients: clients["default"], context_manager_enabled)


def _multi_cluster_runner(runnable, name, context_manager_enabled=False):
    """
    Wrap a runner to pass all cluster clients.

    Args:
        runnable: The runner to wrap
        name: Name for the runner (for repr)
        context_manager_enabled: Whether the runner supports context manager

    Returns:
        A MultiClientRunner that passes all clients
    """
    return MultiClientRunner(runnable, name, lambda clients: clients, context_manager_enabled)


class MultiClientRunner(Runner, Delegator):
    """
    Runner wrapper that handles client extraction from the clients dict.

    This allows runners to be written expecting a single client while
    supporting multi-cluster configurations.
    """

    def __init__(self, runnable, name, client_extractor, context_manager_enabled=False):
        super().__init__(delegate=runnable)
        self.name = name
        self.client_extractor = client_extractor
        self.context_manager_enabled = context_manager_enabled

    async def __call__(self, *args):
        return await self.delegate(self.client_extractor(args[0]), *args[1:])

    def __repr__(self, *args, **kwargs):
        if self.context_manager_enabled:
            return "user-defined context-manager enabled runner for [%s]" % self.name
        else:
            return "user-defined runner for [%s]" % self.name

    async def __aenter__(self):
        if self.context_manager_enabled:
            await self.delegate.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.context_manager_enabled:
            return await self.delegate.__aexit__(exc_type, exc_val, exc_tb)
        else:
            return False


class AssertingRunner(Runner, Delegator):
    """
    Runner wrapper that validates assertions after execution.

    When assertions are enabled (via enable_assertions(True)), this wrapper
    checks assertion conditions specified in the params and raises an error
    if any assertion fails.
    """

    assertions_enabled = False

    def __init__(self, delegate):
        super().__init__(delegate=delegate)
        self.predicates = {
            ">": self.greater_than,
            ">=": self.greater_than_or_equal,
            "<": self.smaller_than,
            "<=": self.smaller_than_or_equal,
            "==": self.equal,
        }

    def greater_than(self, expected, actual):
        return actual > expected

    def greater_than_or_equal(self, expected, actual):
        return actual >= expected

    def smaller_than(self, expected, actual):
        return actual < expected

    def smaller_than_or_equal(self, expected, actual):
        return actual <= expected

    def equal(self, expected, actual):
        return actual == expected

    def check_assertion(self, op_name, assertion, properties):
        path = assertion["property"]
        predicate_name = assertion["condition"]
        expected_value = assertion["value"]
        actual_value = properties
        for k in path.split("."):
            actual_value = actual_value[k]
        predicate = self.predicates[predicate_name]
        success = predicate(expected_value, actual_value)
        if not success:
            if op_name:
                msg = f"Expected [{path}] in [{op_name}] to be {predicate_name} [{expected_value}] but was [{actual_value}]."
            else:
                msg = f"Expected [{path}] to be {predicate_name} [{expected_value}] but was [{actual_value}]."

            raise exceptions.BenchmarkTaskAssertionError(msg)

    async def __call__(self, *args):
        params = args[1]
        return_value = await self.delegate(*args)
        if AssertingRunner.assertions_enabled and "assertions" in params:
            op_name = params.get("name")
            if isinstance(return_value, dict):
                for assertion in params["assertions"]:
                    self.check_assertion(op_name, assertion, return_value)
            else:
                self.logger.debug("Skipping assertion check in [%s] as [%s] does not return a dict.",
                                  op_name, repr(self.delegate))
        return return_value

    def __repr__(self, *args, **kwargs):
        return repr(self.delegate)

    async def __aenter__(self):
        await self.delegate.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self.delegate.__aexit__(exc_type, exc_val, exc_tb)


class NoCompletion(Runner, Delegator):
    """
    Runner wrapper for runners that don't support completion tracking.

    Returns None for completed and task_progress properties.
    """

    def __init__(self, delegate):
        super().__init__(delegate=delegate)

    @property
    def completed(self):
        return None

    @property
    def task_progress(self):
        return None

    async def __call__(self, *args):
        return await self.delegate(*args)

    def __repr__(self, *args, **kwargs):
        return repr(self.delegate)

    async def __aenter__(self):
        await self.delegate.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self.delegate.__aexit__(exc_type, exc_val, exc_tb)


class WithCompletion(Runner, Delegator):
    """
    Runner wrapper that tracks task completion progress.

    Delegates completed and task_progress properties to a progressable target.
    """

    def __init__(self, delegate, progressable):
        super().__init__(delegate=delegate)
        self.progressable = progressable

    @property
    def completed(self):
        return self.progressable.completed

    @property
    def task_progress(self):
        return self.progressable.task_progress

    async def __call__(self, *args):
        return await self.delegate(*args)

    def __repr__(self, *args, **kwargs):
        return repr(self.delegate)

    async def __aenter__(self):
        await self.delegate.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self.delegate.__aexit__(exc_type, exc_val, exc_tb)


def _with_assertions(delegate):
    """Wrap a runner with assertion checking."""
    return AssertingRunner(delegate)


def _with_completion(delegate):
    """
    Wrap a runner with completion tracking if the runner supports it.

    Args:
        delegate: The runner to wrap

    Returns:
        WithCompletion if the runner has completed/task_progress properties,
        NoCompletion otherwise
    """
    unwrapped_runner = unwrap(delegate)
    if hasattr(unwrapped_runner, "completed") and hasattr(unwrapped_runner, "task_progress"):
        return WithCompletion(delegate, unwrapped_runner)
    else:
        return NoCompletion(delegate)


def wrap_runner_for_registration(runner, operation_type, async_runner):
    """
    Apply standard wrappers to a runner during registration.

    This is the common logic for wrapping runners with cluster-awareness,
    assertions, and completion tracking.

    Args:
        runner: The runner instance to wrap
        operation_type: The operation type being registered
        async_runner: Must be True (sync runners are not supported)

    Returns:
        The fully wrapped runner

    Raises:
        BenchmarkAssertionError: If async_runner is False
    """
    logger = logging.getLogger(__name__)

    if not async_runner:
        raise exceptions.BenchmarkAssertionError(
            "Runner [{}] must be implemented as async runner and registered with async_runner=True.".format(str(runner)))

    if getattr(runner, "multi_cluster", False):
        if "__aenter__" in dir(runner) and "__aexit__" in dir(runner):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Registering runner object [%s] for [%s].", str(runner), str(operation_type))
            cluster_aware_runner = _multi_cluster_runner(runner, str(runner), context_manager_enabled=True)
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Registering context-manager capable runner object [%s] for [%s].", str(runner), str(operation_type))
            cluster_aware_runner = _multi_cluster_runner(runner, str(runner))
    elif isinstance(runner, types.FunctionType):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Registering runner function [%s] for [%s].", str(runner), str(operation_type))
        cluster_aware_runner = _single_cluster_runner(runner, runner.__name__)
    elif "__aenter__" in dir(runner) and "__aexit__" in dir(runner):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Registering context-manager capable runner object [%s] for [%s].", str(runner), str(operation_type))
        cluster_aware_runner = _single_cluster_runner(runner, str(runner), context_manager_enabled=True)
    else:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Registering runner object [%s] for [%s].", str(runner), str(operation_type))
        cluster_aware_runner = _single_cluster_runner(runner, str(runner))

    return _with_completion(_with_assertions(cluster_aware_runner))
