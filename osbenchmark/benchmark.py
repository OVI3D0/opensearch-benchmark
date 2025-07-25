# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.
# Licensed to Elasticsearch B.V. under one or more contributor
# license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright
# ownership. Elasticsearch B.V. licenses this file to you under
# the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import argparse
import datetime
import logging
import os
import platform
import sys
import time
import uuid

import thespian.actors

from osbenchmark import PROGRAM_NAME, BANNER, FORUM_LINK, SKULL, check_python_version, doc_link, telemetry
from osbenchmark import version, actor, config, paths, \
    test_execution_orchestrator, results_publisher, \
        metrics, workload, exceptions, log
from osbenchmark.builder import provision_config, builder
from osbenchmark.workload_generator import workload_generator
from osbenchmark.utils import io, convert, process, console, net, opts, versions
from osbenchmark import aggregator
from osbenchmark.worker_coordinator.worker_coordinator import ConfigureFeedbackScaling

def create_arg_parser():
    def positive_number(v):
        value = int(v)
        if value <= 0:
            raise argparse.ArgumentTypeError(f"must be positive but was {value}")
        return value

    def non_empty_list(arg):
        lst = opts.csv_to_list(arg)
        if len(lst) < 1:
            raise argparse.ArgumentError(argument=None, message="At least one argument required!")
        return lst

    def runtime_jdk(v):
        if v == "bundled":
            return v
        else:
            try:
                return positive_number(v)
            except argparse.ArgumentTypeError:
                raise argparse.ArgumentTypeError(f"must be a positive number or 'bundled' but was {v}")

    def supported_os_version(v):
        if v:
            min_os_version = versions.Version.from_string(version.minimum_os_version())
            specified_version = versions.Version.from_string(v)
            if specified_version < min_os_version:
                raise argparse.ArgumentTypeError(f"must be at least {min_os_version} but was {v}")
        return v

    def add_workload_source(subparser):
        workload_source_group = subparser.add_mutually_exclusive_group()
        workload_source_group.add_argument(
            "--workload-repository",
            help="Define the repository from where OSB will load workloads (default: default).",
            # argparse is smart enough to use this default only if the user did not use --workload-path and also did not specify anything
            default="default"
        )
        workload_source_group.add_argument(
            "--workload-path",
            help="Define the path to a workload.")
        subparser.add_argument(
            "--workload-revision",
            help="Define a specific revision in the workload repository that OSB should use.",
            default=None)

    # try to preload configurable defaults, but this does not work together with `--configuration-name` (which is undocumented anyway)
    cfg = config.Config()
    if cfg.config_present():
        cfg.load_config()
        preserve_install = cfg.opts("defaults", "preserve_benchmark_candidate", default_value=False, mandatory=False)
    else:
        preserve_install = False

    parser = argparse.ArgumentParser(prog=PROGRAM_NAME,
                                     description=BANNER + "\n\n A benchmarking tool for OpenSearch",
                                     epilog="Find out more about OSB at {}".format(console.format.link(doc_link())),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version="%(prog)s " + version.version())

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    subparsers = parser.add_subparsers(
        title="subcommands",
        dest="subcommand",
        help="")

    test_execution_parser = subparsers.add_parser("execute-test", help="Run a benchmark")
    # change in favor of "list telemetry", "list workloads", "list pipelines"
    list_parser = subparsers.add_parser("list", help="List configuration options")
    list_parser.add_argument(
        "configuration",
        metavar="configuration",
        help="The configuration for which OSB should show the available options. "
             "Possible values are: telemetry, workloads, pipelines, test_executions, provision_config_instances, opensearch-plugins",
        choices=["telemetry", "workloads", "pipelines", "test_executions", "aggregated_results",
                 "provision_config_instances", "opensearch-plugins"])
    list_parser.add_argument(
        "--limit",
        help="Limit the number of search results for recent test_executions (default: 10).",
        default=10,
    )
    add_workload_source(list_parser)

    info_parser = subparsers.add_parser("info", help="Show info about a workload")
    add_workload_source(info_parser)
    info_parser.add_argument(
        "--workload",
        "-w",
        help=f"Define the workload to use. List possible workloads with `{PROGRAM_NAME} list workloads`."
        # we set the default value later on because we need to determine whether the user has provided this value.
        # default="geonames"
    )

    info_parser.add_argument(
        "--workload-params",
        "-wp",
        help="Define a comma-separated list of key:value pairs that are injected verbatim to the workload as variables.",
        default=""
    )
    info_parser.add_argument(
        "--test-procedure",
        help=f"Define the test_procedure to use. List possible test_procedures for workloads with `{PROGRAM_NAME} list workloads`."
    )
    info_task_filter_group = info_parser.add_mutually_exclusive_group()
    info_task_filter_group.add_argument(
        "--include-tasks",
        help="Defines a comma-separated list of tasks to run. By default all tasks of a test_procedure are run.")
    info_task_filter_group.add_argument(
        "--exclude-tasks",
        help="Defines a comma-separated list of tasks not to run. By default all tasks of a test_procedure are run.")

    create_workload_parser = subparsers.add_parser("create-workload", help="Create a OSB workload from existing data")
    create_workload_parser.add_argument(
        "--workload",
        "-w",
        required=True,
        help="Name of the generated workload")
    create_workload_parser.add_argument(
        "--indices",
        "-i",
        type=non_empty_list,
        required=True,
        help="Comma-separated list of indices to include in the workload")
    create_workload_parser.add_argument(
        "--target-hosts",
        "-t",
        default="",
        required=True,
        help="Comma-separated list of host:port pairs which should be targeted")
    create_workload_parser.add_argument(
        "--client-options",
        "-c",
        default=opts.ClientOptions.DEFAULT_CLIENT_OPTIONS,
        help=f"Comma-separated list of client options to use. (default: {opts.ClientOptions.DEFAULT_CLIENT_OPTIONS})")
    create_workload_parser.add_argument(
        "--output-path",
        default=os.path.join(os.getcwd(), "workloads"),
        help="Workload output directory (default: workloads/)")
    create_workload_parser.add_argument(
        "--custom-queries",
        type=argparse.FileType('r'),
        help="Input JSON file to use containing custom workload queries that override the default match_all query")
    create_workload_parser.add_argument(
        "--number-of-docs",
        action=opts.StoreKeyPairAsDict,
        nargs='+',
        metavar="KEY:VAL",
        help="Map of index name and integer doc count to extract. Ensure that index name also exists in --indices parameter. " +
        "To specify several indices and doc counts, use format: <index1>:<doc_count1> <index2>:<doc_count2> ...")
    create_workload_parser.add_argument(
        "--sample-frequency",
        action=opts.StoreKeyPairAsDict,
        nargs='+',
        metavar="KEY:VAL",
        help="Map of index name and an integer, representing the sample frequency of docs that should be extracted per index. " +
        "Ensure that index name also exists in --indices parameter. " +
        "To specify several indices and doc counts, use format: <index1>:<sample-frequency-1> <index2>:<sample-frequency-2> ...")

    compare_parser = subparsers.add_parser("compare", help="Compare two test_executions")
    compare_parser.add_argument(
        "--baseline",
        "-b",
        required=True,
        help=f"TestExecution ID of the baseline (see {PROGRAM_NAME} list test_executions).")
    compare_parser.add_argument(
        "--contender",
        "-c",
        required=True,
        help=f"TestExecution ID of the contender (see {PROGRAM_NAME} list test_executions).")
    compare_parser.add_argument(
        "--percentiles",
        help=f"A comma-separated list of percentiles to report latency and service time."
             f"(default: {metrics.GlobalStatsCalculator.DEFAULT_LATENCY_PERCENTILES}).",
        default=metrics.GlobalStatsCalculator.DEFAULT_LATENCY_PERCENTILES)
    compare_parser.add_argument(
        "--results-format",
        help="Define the output format for the command line results (default: markdown).",
        choices=["markdown", "csv"],
        default="markdown")
    compare_parser.add_argument(
        "--results-numbers-align",
        help="Define the output column number alignment for the command line results (default: right).",
        choices=["right", "center", "left", "decimal"],
        default="right")
    compare_parser.add_argument(
        "--results-file",
        help="Write the command line results also to the provided file.",
        default="")
    compare_parser.add_argument(
        "--show-in-results",
        help="Whether to include the comparison in the results file.",
        default=True)

    aggregate_parser = subparsers.add_parser("aggregate", help="Aggregate multiple test_executions")
    aggregate_parser.add_argument(
        "--test-executions",
        type=non_empty_list,
        required=True,
        help="Comma-separated list of TestExecution IDs to aggregate")
    aggregate_parser.add_argument(
        "--test-execution-id",
        "-tid",
        help="Define a unique id for this aggregated test_execution.",
        default="")
    aggregate_parser.add_argument(
        "--results-file",
        help="Write the aggregated results to the provided file.",
        default="")
    aggregate_parser.add_argument(
        "--workload-repository",
        help="Define the repository from where OSB will load workloads (default: default).",
        default="default")

    download_parser = subparsers.add_parser("download", help="Downloads an artifact")
    download_parser.add_argument(
        "--provision-config-repository",
        help="Define the repository from where OSB will load provision_configs and provision_config_instances (default: default).",
        default="default")
    download_parser.add_argument(
        "--provision-config-revision",
        help="Define a specific revision in the provision_config repository that OSB should use.",
        default=None)
    download_parser.add_argument(
        "--provision-config-path",
        help="Define the path to the provision_config_instance and plugin configurations to use.")
    download_parser.add_argument(
        "--distribution-version",
        type=supported_os_version,
        help="Define the version of the OpenSearch distribution to download. "
             "Check https://opensearch.org/docs/version-history/ for released versions.",
        default="")
    download_parser.add_argument(
        "--distribution-repository",
        help="Define the repository from where the OpenSearch distribution should be downloaded (default: release).",
        default="release")
    download_parser.add_argument(
        "--provision-config-instance",
        help=f"Define the provision_config_instance to use. List possible "
        f"provision_config_instances with `{PROGRAM_NAME} list "
        f"provision_config_instances` (default: defaults).",
        default="defaults")  # optimized for local usage
    download_parser.add_argument(
        "--provision-config-instance-params",
        help="Define a comma-separated list of key:value pairs that are injected verbatim as variables for the provision_config_instance.",
        default=""
    )
    download_parser.add_argument(
        "--target-os",
        help="The name of the target operating system for which an artifact should be downloaded (default: current OS)",
    )
    download_parser.add_argument(
        "--target-arch",
        help="The name of the CPU architecture for which an artifact should be downloaded (default: current architecture)",
    )

    install_parser = subparsers.add_parser("install", help="Installs an OpenSearch node locally")
    install_parser.add_argument(
        "--revision",
        help="Define the source code revision for building the benchmark candidate. 'current' uses the source tree as is,"
             " 'latest' fetches the latest version on main. It is also possible to specify a commit id or a timestamp."
             " The timestamp must be specified as: \"@ts\" where \"ts\" must be a valid ISO 8601 timestamp, "
             "e.g. \"@2013-07-27T10:37:00Z\" (default: current).",
        default="current")  # optimized for local usage, don't fetch sources
    # Intentionally undocumented as we do not consider Docker a fully supported option.
    install_parser.add_argument(
        "--build-type",
        help=argparse.SUPPRESS,
        choices=["tar", "docker"],
        default="tar")
    install_parser.add_argument(
        "--provision-config-repository",
        help="Define the repository from where OSB will load provision_configs and provision_config_instances (default: default).",
        default="default")
    install_parser.add_argument(
        "--provision-config-revision",
        help="Define a specific revision in the provision_config repository that OSB should use.",
        default=None)
    install_parser.add_argument(
        "--provision-config-path",
        help="Define the path to the provision_config_instance and plugin configurations to use.")
    install_parser.add_argument(
        "--runtime-jdk",
        type=runtime_jdk,
        help="The major version of the runtime JDK to use during installation.",
        default=None)
    install_parser.add_argument(
        "--distribution-repository",
        help="Define the repository from where the OpenSearch distribution should be downloaded (default: release).",
        default="release")
    install_parser.add_argument(
        "--distribution-version",
        type=supported_os_version,
        help="Define the version of the OpenSearch distribution to download. "
             "Check https://opensearch.org/docs/version-history/ for released versions.",
        default="")
    install_parser.add_argument(
        "--provision-config-instance",
        help=f"Define the provision_config_instance to use. List possible "
        f"provision_config_instances with `{PROGRAM_NAME} list "
        f"provision_config_instances` (default: defaults).",
        default="defaults")  # optimized for local usage
    install_parser.add_argument(
        "--provision-config-instance-params",
        help="Define a comma-separated list of key:value pairs that are injected verbatim as variables for the provision_config_instance.",
        default=""
    )
    install_parser.add_argument(
        "--opensearch-plugins",
        help="Define the OpenSearch plugins to install. (default: install no plugins).",
        default="")
    install_parser.add_argument(
        "--plugin-params",
        help="Define a comma-separated list of key:value pairs that are injected verbatim to all plugins as variables.",
        default=""
    )
    install_parser.add_argument(
        "--network-host",
        help="The IP address to bind to and publish",
        default="127.0.0.1"
    )
    install_parser.add_argument(
        "--http-port",
        help="The port to expose for HTTP traffic",
        default="39200"
    )
    install_parser.add_argument(
        "--node-name",
        help="The name of this OpenSearch node",
        default="benchmark-node-0"
    )
    install_parser.add_argument(
        "--master-nodes",
        help="A comma-separated list of the initial master node names",
        default=""
    )
    install_parser.add_argument(
        "--seed-hosts",
        help="A comma-separated list of the initial seed host IPs",
        default=""
    )

    start_parser = subparsers.add_parser("start", help="Starts an OpenSearch node locally")
    start_parser.add_argument(
        "--installation-id",
        required=True,
        help="The id of the installation to start",
        # the default will be dynamically derived by
        # test_execution_orchestrator based on the
        # presence / absence of other command line options
        default="")
    start_parser.add_argument(
        "--test-execution-id",
        "-tid",
        required=True,
        help="Define a unique id for this test_execution.",
        default="")
    start_parser.add_argument(
        "--runtime-jdk",
        type=runtime_jdk,
        help="The major version of the runtime JDK to use.",
        default=None)
    start_parser.add_argument(
        "--telemetry",
        help=f"Enable the provided telemetry devices, provided as a comma-separated list. List possible telemetry "
             f"devices with `{PROGRAM_NAME} list telemetry`.",
        default="")
    start_parser.add_argument(
        "--telemetry-params",
        help="Define a comma-separated list of key:value pairs that are injected verbatim to the telemetry devices as parameters.",
        default=""
    )

    stop_parser = subparsers.add_parser("stop", help="Stops an OpenSearch node locally")
    stop_parser.add_argument(
        "--installation-id",
        required=True,
        help="The id of the installation to stop",
        # the default will be dynamically derived by
        # test_execution_orchestrator based on the
        # presence / absence of other command line options
        default="")
    stop_parser.add_argument(
        "--preserve-install",
        help=f"Keep the benchmark candidate and its index. (default: {str(preserve_install).lower()}).",
        default=preserve_install,
        action="store_true")

    for p in [list_parser, test_execution_parser]:
        p.add_argument(
            "--distribution-version",
            type=supported_os_version,
            help="Define the version of the OpenSearch distribution to download. "
                 "Check https://opensearch.org/docs/version-history/ for released versions.",
            default="")
        p.add_argument(
            "--provision-config-path",
            help="Define the path to the provision_config_instance and plugin configurations to use.")
        p.add_argument(
            "--provision-config-repository",
            help="Define repository from where OSB will load provision_configs and provision_config_instances (default: default).",
            default="default")
        p.add_argument(
            "--provision-config-revision",
            help="Define a specific revision in the provision_config repository that OSB should use.",
            default=None)

    test_execution_parser.add_argument(
        "--test-execution-id",
        "-tid",
        help="Define a unique id for this test_execution.",
        default=str(uuid.uuid4()))
    test_execution_parser.add_argument(
        "--pipeline",
        help="Select the pipeline to run.",
        # the default will be dynamically derived by
        # test_execution_orchestrator based on the
        # presence / absence of other command line options
        default="")
    test_execution_parser.add_argument(
        "--revision",
        help="Define the source code revision for building the benchmark candidate. 'current' uses the source tree as is,"
             " 'latest' fetches the latest version on main. It is also possible to specify a commit id or a timestamp."
             " The timestamp must be specified as: \"@ts\" where \"ts\" must be a valid ISO 8601 timestamp, "
             "e.g. \"@2013-07-27T10:37:00Z\" (default: current).",
        default="current")  # optimized for local usage, don't fetch sources
    add_workload_source(test_execution_parser)
    test_execution_parser.add_argument(
        "--workload",
        "-w",
        help=f"Define the workload to use. List possible workloads with `{PROGRAM_NAME} list workloads`."
    )
    test_execution_parser.add_argument(
        "--workload-params",
        "-wp",
        help="Define a comma-separated list of key:value pairs that are injected verbatim to the workload as variables.",
        default=""
    )
    test_execution_parser.add_argument(
        "--test-procedure",
        help=f"Define the test_procedure to use. List possible test_procedures for workloads with `{PROGRAM_NAME} list workloads`.")
    test_execution_parser.add_argument(
        "--provision-config-instance",
        help=f"Define the provision_config_instance to use. List possible "
        f"provision_config_instances with `{PROGRAM_NAME} list "
        f"provision_config_instances` (default: defaults).",
        default="defaults")  # optimized for local usage
    test_execution_parser.add_argument(
        "--provision-config-instance-params",
        help="Define a comma-separated list of key:value pairs that are injected verbatim as variables for the provision_config_instance.",
        default=""
    )
    test_execution_parser.add_argument(
        "--runtime-jdk",
        type=runtime_jdk,
        help="The major version of the runtime JDK to use.",
        default=None)
    test_execution_parser.add_argument(
        "--opensearch-plugins",
        help="Define the OpenSearch plugins to install. (default: install no plugins).",
        default="")
    test_execution_parser.add_argument(
        "--plugin-params",
        help="Define a comma-separated list of key:value pairs that are injected verbatim to all plugins as variables.",
        default=""
    )
    test_execution_parser.add_argument(
        "--target-hosts",
        "-t",
        help="Define a comma-separated list of host:port pairs which should be targeted if using the pipeline 'benchmark-only' "
             "(default: localhost:9200).",
        default="")  # actually the default is pipeline specific and it is set later
    test_execution_parser.add_argument(
        "--load-worker-coordinator-hosts",
        help="Define a comma-separated list of hosts which should generate load (default: localhost).",
        default="localhost")
    test_execution_parser.add_argument(
        "--client-options",
        "-c",
        help=f"Define a comma-separated list of client options to use. The options will be passed to the OpenSearch "
             f"Python client (default: {opts.ClientOptions.DEFAULT_CLIENT_OPTIONS}).",
        default=opts.ClientOptions.DEFAULT_CLIENT_OPTIONS)
    test_execution_parser.add_argument("--on-error",
                             choices=["continue", "abort"],
                             help="Controls how OSB behaves on response errors (default: continue).",
                             default="continue")
    test_execution_parser.add_argument(
        "--telemetry",
        help=f"Enable the provided telemetry devices, provided as a comma-separated list. List possible telemetry "
             f"devices with `{PROGRAM_NAME} list telemetry`.",
        default="")
    test_execution_parser.add_argument(
        "--telemetry-params",
        help="Define a comma-separated list of key:value pairs that are injected verbatim to the telemetry devices as parameters.",
        default=""
    )
    test_execution_parser.add_argument(
        "--distribution-repository",
        help="Define the repository from where the OpenSearch distribution should be downloaded (default: release).",
        default="release")

    task_filter_group = test_execution_parser.add_mutually_exclusive_group()
    task_filter_group.add_argument(
        "--include-tasks",
        help="Defines a comma-separated list of tasks to run. By default all tasks of a test_procedure are run.")
    task_filter_group.add_argument(
        "--exclude-tasks",
        help="Defines a comma-separated list of tasks not to run. By default all tasks of a test_procedure are run.")
    test_execution_parser.add_argument(
        "--user-tag",
        help="Define a user-specific key-value pair (separated by ':'). It is added to each metric record as meta info. "
             "Example: intention:baseline-ticket-12345",
        default="")
    test_execution_parser.add_argument(
        "--results-format",
        help="Define the output format for the command line results (default: markdown).",
        choices=["markdown", "csv"],
        default="markdown")
    test_execution_parser.add_argument(
        "--results-numbers-align",
        help="Define the output column number alignment for the command line results (default: right).",
        choices=["right", "center", "left", "decimal"],
        default="right")
    test_execution_parser.add_argument(
        "--show-in-results",
        help="Define which values are shown in the summary results published (default: available).",
        choices=["available", "all-percentiles", "all"],
        default="available")
    test_execution_parser.add_argument(
        "--results-file",
        help="Write the command line results also to the provided file.",
        default="")
    test_execution_parser.add_argument(
        "--preserve-install",
        help=f"Keep the benchmark candidate and its index. (default: {str(preserve_install).lower()}).",
        default=preserve_install,
        action="store_true")
    test_execution_parser.add_argument(
        "--test-mode",
        help="Runs the given workload in 'test mode'. Meant to check a workload for errors but not for real benchmarks (default: false).",
        default=False,
        action="store_true")
    test_execution_parser.add_argument(
        "--enable-worker-coordinator-profiling",
        help="Enables a profiler for analyzing the performance of calls in OSB's worker coordinator (default: false).",
        default=False,
        action="store_true")
    test_execution_parser.add_argument(
        "--enable-assertions",
        help="Enables assertion checks for tasks (default: false).",
        default=False,
        action="store_true")
    test_execution_parser.add_argument(
        "--kill-running-processes",
        "-k",
        action="store_true",
        default=False,
        help="If any processes is running, it is going to kill them and allow OSB to continue to run."
    )
    test_execution_parser.add_argument(
        "--latency-percentiles",
        help=f"A comma-separated list of percentiles to report for latency "
             f"(default: {metrics.GlobalStatsCalculator.DEFAULT_LATENCY_PERCENTILES}).",
        default=metrics.GlobalStatsCalculator.DEFAULT_LATENCY_PERCENTILES
    )
    test_execution_parser.add_argument(
        "--throughput-percentiles",
        help=f"A comma-separated list of percentiles to report for throughput, in addition to mean/median/max/min "
             f"(default: {metrics.GlobalStatsCalculator.DEFAULT_THROUGHPUT_PERCENTILES}).",
        default=metrics.GlobalStatsCalculator.DEFAULT_THROUGHPUT_PERCENTILES
    )
    test_execution_parser.add_argument(
        "--randomization-enabled",
        help="Runs the given workload with query randomization enabled (default: false).",
        default=False,
        action="store_true")
    test_execution_parser.add_argument(
        "--randomization-repeat-frequency",
        help=f"The repeat_frequency for query randomization. Ignored if randomization is off"
             f"(default: {workload.loader.QueryRandomizerWorkloadProcessor.DEFAULT_RF}).",
        default=workload.loader.QueryRandomizerWorkloadProcessor.DEFAULT_RF)
    test_execution_parser.add_argument(
        "--randomization-n",
        help=f"The number of standard values to generate for each field for query randomization."
             f"Ignored if randomization is off (default: {workload.loader.QueryRandomizerWorkloadProcessor.DEFAULT_N}).",
        default=workload.loader.QueryRandomizerWorkloadProcessor.DEFAULT_N)
    test_execution_parser.add_argument(
        "--randomization-alpha",
        help=f"The alpha parameter used for the Zipf distribution for query randomization. Low values spread the distribution out, "
             f"high values favor the most common queries. "
             f"Ignored if randomization is off (default: {workload.loader.QueryRandomizerWorkloadProcessor.DEFAULT_ALPHA}).",
        default=workload.loader.QueryRandomizerWorkloadProcessor.DEFAULT_ALPHA)
    test_execution_parser.add_argument(
        "--test-iterations",
        help="The number of times to run the workload (default: 1).",
        default=1)
    test_execution_parser.add_argument(
        "--aggregate",
        type=lambda x: (str(x).lower() in ['true', '1', 'yes', 'y']),
        help="Aggregate the results of multiple test executions (default: true).",
        default=True)
    test_execution_parser.add_argument(
        "--sleep-timer",
        help="Sleep for the specified number of seconds before starting the next test execution (default: 5).",
        default=5)
    test_execution_parser.add_argument(
        "--cancel-on-error",
        action="store_true",
        help="Stop executing tests if an error occurs in one of the test iterations (default: false).",
    )
    test_execution_parser.add_argument(
        "--load-test-qps",
        help="Run a load test on your cluster, up to a certain QPS value (default: 0)",
        default=0
    )
    test_execution_parser.add_argument(
        "--redline-test",
        help="Run a redline test on your cluster, up to a certain QPS value (default: 1000)",
        nargs='?',
        const=1000,  # Value to use when flag is present but no value given
        default=0,  # Value to use when flag is not present
        type=int
    )
    test_execution_parser.add_argument(
        "--redline-scale-step",
        type=int,
        help="How many clients to add while scaling up during redline testing (default: 5).",
        default=ConfigureFeedbackScaling.DEFAULT_SCALE_STEP
    )
    test_execution_parser.add_argument(
        "--redline-scaledown-percentage",
        type=float,
        help="What percentage of clients to remove when errors occur (default: 10%%).",
        default=ConfigureFeedbackScaling.DEFAULT_SCALEDOWN_PCT
    )
    test_execution_parser.add_argument(
        "--redline-post-scaledown-sleep",
        type=int,
        help="How many seconds to wait before scaling up again after a scale down (default: 30).",
        default=ConfigureFeedbackScaling.DEFAULT_SLEEP_SECONDS
    )
    test_execution_parser.add_argument(
        "--redline-max-clients",
        type=int,
        help="Maximum number of clients to allow during redline testing. If not set, will default to clients defined in the test procedure.",
        default=None
    )
    test_execution_parser.add_argument(
        "--redline-max-cpu-usage",
        type=int,
        help="Maximum CPU utilization before scaling back client numbers. Used to activate CPU-based feedback in OSB.",
        default=None
    )
    test_execution_parser.add_argument(
        "--redline-cpu-window-seconds",
        type=int,
        help="How many seconds the window for average CPU load should be in seconds during CPU-based redline testing. (Default: 30)",
        default=ConfigureFeedbackScaling.DEFAULT_CPU_WINDOW_SECONDS
    )
    test_execution_parser.add_argument(
        "--redline-cpu-check-interval",
        type=int,
        help="How many seconds between CPU checks there should be during CPU-based redline testing. (Default: 30)",
        default=ConfigureFeedbackScaling.DEFAULT_CPU_CHECK_INTERVAL
    )

    ###############################################################################
    #
    # The options below are undocumented and can be removed or changed at any time.
    #
    ###############################################################################
    # This option is intended to tell OSB to assume a different start date than 'now'. This is effectively just useful for things like
    # backtesting or a benchmark run across environments (think: comparison of EC2 and bare metal) but never for the typical user.
    test_execution_parser.add_argument(
        "--effective-start-date",
        help=argparse.SUPPRESS,
        type=lambda s: datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S"),
        default=None)
    # Skips checking that the REST API is available before proceeding with the benchmark
    test_execution_parser.add_argument(
        "--skip-rest-api-check",
        help=argparse.SUPPRESS,
        action="store_true",
        default=False)

    for p in [list_parser, test_execution_parser, compare_parser, aggregate_parser,
              download_parser, install_parser, start_parser, stop_parser, info_parser, create_workload_parser]:
        # This option is needed to support a separate configuration for the integration tests on the same machine
        p.add_argument(
            "--configuration-name",
            help=argparse.SUPPRESS,
            default=None)
        p.add_argument(
            "--quiet",
            help="Suppress as much as output as possible (default: false).",
            default=False,
            action="store_true")
        p.add_argument(
            "--offline",
            help="Assume that OSB has no connection to the Internet (default: false).",
            default=False,
            action="store_true")

    return parser


def dispatch_list(cfg):
    what = cfg.opts("system", "list.config.option")
    if what == "telemetry":
        telemetry.list_telemetry()
    elif what == "workloads":
        workload.list_workloads(cfg)
    elif what == "pipelines":
        test_execution_orchestrator.list_pipelines()
    elif what == "test_executions":
        metrics.list_test_executions(cfg)
    elif what == "aggregated_results":
        metrics.list_aggregated_results(cfg)
    elif what == "provision_config_instances":
        provision_config.list_provision_config_instances(cfg)
    elif what == "opensearch-plugins":
        provision_config.list_plugins(cfg)
    else:
        raise exceptions.SystemSetupError("Cannot list unknown configuration option [%s]" % what)


def print_help_on_errors():
    heading = "Getting further help:"
    console.println(console.format.bold(heading))
    console.println(console.format.underline_for(heading))
    console.println(f"* Check the log files in {paths.logs()} for errors.")
    console.println(f"* Read the documentation at {console.format.link(doc_link())}.")
    console.println(f"* Ask a question on the forum at {console.format.link(FORUM_LINK)}.")
    console.println(f"* Raise an issue at {console.format.link('https://github.com/opensearch-project/OpenSearch-Benchmark/issues')} "
                    f"and include the log files in {paths.logs()}.")


def execute_test(cfg, kill_running_processes=False):
    logger = logging.getLogger(__name__)

    if kill_running_processes:
        logger.info("Killing running OSB processes")

        # Kill any lingering OSB processes before attempting to continue - the actor system needs to be a singleton on this machine
        # noinspection PyBroadException
        try:
            process.kill_running_benchmark_instances()
        except BaseException:
            logger.exception(
                "Could not terminate potentially running OSB instances correctly. Attempting to go on anyway.")
    else:
        other_benchmark_processes = process.find_all_other_benchmark_processes()
        if other_benchmark_processes:
            pids = [p.pid for p in other_benchmark_processes]

            msg = f"There are other OSB processes running on this machine (PIDs: {pids}) but only one OSB " \
                  f"benchmark is allowed to run at the same time.\n\nYou can use --kill-running-processes flag " \
                  f"to kill running processes automatically and allow OSB to continue to run a new benchmark. " \
                  f"Otherwise, you need to manually kill them."
            raise exceptions.BenchmarkError(msg)

    # redline testing: check metrics store type before running cpu based feedback test
    cpu_max = cfg.opts("workload", "redline.max_cpu_usage", default_value=None, mandatory=False)
    if cpu_max is not None:
        store = metrics.metrics_store(cfg, read_only=False)
        try:
            if isinstance(store, metrics.InMemoryMetricsStore):
                raise exceptions.SystemSetupError(
                    "CPU-based feedback requires a metrics store, but you're using the in-memory store. "
                    "Specify a metrics store in your benchmark.ini or via CLI to continue."
                )
        finally:
            store.close()

    with_actor_system(test_execution_orchestrator.run, cfg)


def with_actor_system(runnable, cfg):
    logger = logging.getLogger(__name__)
    already_running = actor.actor_system_already_running()
    logger.info("Actor system already running locally? [%s]", str(already_running))
    try:
        actors = actor.bootstrap_actor_system(try_join=already_running, prefer_local_only=not already_running)
        # We can only support remote benchmarks if we have a dedicated daemon that is not only bound to 127.0.0.1
        cfg.add(config.Scope.application, "system", "remote.benchmarking.supported", already_running)
    # This happens when the admin process could not be started, e.g. because it could not open a socket.
    except thespian.actors.InvalidActorAddress:
        logger.info("Falling back to offline actor system.")
        actor.use_offline_actor_system()
        actors = actor.bootstrap_actor_system(try_join=True)
    except Exception as e:
        logger.exception("Could not bootstrap actor system.")
        if str(e) == "Unable to determine valid external socket address.":
            console.warn("Could not determine a socket address. Are you running without any network? Switching to degraded mode.",
                         logger=logger)
            logger.info("Falling back to offline actor system.")
            actor.use_offline_actor_system()
            actors = actor.bootstrap_actor_system(try_join=True)
        else:
            raise
    try:
        runnable(cfg)
    finally:
        # We only shutdown the actor system if it was not already running before
        if not already_running:
            shutdown_complete = False
            times_interrupted = 0
            # give some time for any outstanding messages to be delivered to the actor system
            time.sleep(3)
            while not shutdown_complete and times_interrupted < 2:
                try:
                    logger.info("Attempting to shutdown internal actor system.")
                    actors.shutdown()
                    # note that this check will only evaluate to True for a TCP-based actor system.
                    timeout = 15
                    while actor.actor_system_already_running() and timeout > 0:
                        logger.info("Actor system is still running. Waiting...")
                        time.sleep(1)
                        timeout -= 1
                    if timeout > 0:
                        shutdown_complete = True
                        logger.info("Shutdown completed.")
                    else:
                        logger.warning("Shutdown timed out. Actor system is still running.")
                        break
                except KeyboardInterrupt:
                    times_interrupted += 1
                    logger.warning("User interrupted shutdown of internal actor system.")
                    console.info("Please wait a moment for OSB's internal components to shutdown.")
            if not shutdown_complete and times_interrupted > 0:
                logger.warning("Terminating after user has interrupted actor system shutdown explicitly for [%d] times.",
                               times_interrupted)
                console.println("")
                console.warn("Terminating now at the risk of leaving child processes behind.")
                console.println("")
                console.warn("The next test_execution may fail due to an unclean shutdown.")
                console.println("")
                console.println(SKULL)
                console.println("")
            elif not shutdown_complete:
                console.warn("Could not terminate all internal processes within timeout. Please check and force-terminate "
                             "all OSB processes.")



def configure_telemetry_params(args, cfg):
    cfg.add(config.Scope.applicationOverride, "telemetry", "devices", opts.csv_to_list(args.telemetry))
    cfg.add(config.Scope.applicationOverride, "telemetry", "params", opts.to_dict(args.telemetry_params))


def configure_workload_params(arg_parser, args, cfg, command_requires_workload=True):
    cfg.add(config.Scope.applicationOverride, "workload", "repository.revision", args.workload_revision)
    # We can assume here that if a workload-path is given, the user did not specify a repository either (although argparse sets it to
    # its default value)
    if args.workload_path:
        cfg.add(config.Scope.applicationOverride, "workload", "workload.path", os.path.abspath(io.normalize_path(args.workload_path)))
        cfg.add(config.Scope.applicationOverride, "workload", "repository.name", None)
        if args.workload_revision:
            # stay as close as possible to argparse errors although we have a custom validation.
            arg_parser.error("argument --workload-revision not allowed with argument --workload-path")
        if command_requires_workload and args.workload:
            # stay as close as possible to argparse errors although we have a custom validation.
            arg_parser.error("argument --workload not allowed with argument --workload-path")
    else:
        cfg.add(config.Scope.applicationOverride, "workload", "repository.name", args.workload_repository)
        if command_requires_workload:
            if not args.workload:
                raise arg_parser.error("argument --workload is required")
            cfg.add(config.Scope.applicationOverride, "workload", "workload.name", args.workload)

    if command_requires_workload:
        cfg.add(config.Scope.applicationOverride, "workload", "params", opts.to_dict(args.workload_params))
        cfg.add(config.Scope.applicationOverride, "workload", "test_procedure.name", args.test_procedure)
        cfg.add(config.Scope.applicationOverride, "workload", "include.tasks", opts.csv_to_list(args.include_tasks))
        cfg.add(config.Scope.applicationOverride, "workload", "exclude.tasks", opts.csv_to_list(args.exclude_tasks))


def configure_builder_params(args, cfg, command_requires_provision_config_instance=True):
    if args.provision_config_path:
        cfg.add(
            config.Scope.applicationOverride, "builder",
            "provision_config.path", os.path.abspath(
                io.normalize_path(args.provision_config_path)))
        cfg.add(config.Scope.applicationOverride, "builder", "repository.name", None)
        cfg.add(config.Scope.applicationOverride, "builder", "repository.revision", None)
    else:
        cfg.add(config.Scope.applicationOverride, "builder", "repository.name", args.provision_config_repository)
        cfg.add(config.Scope.applicationOverride, "builder", "repository.revision", args.provision_config_revision)

    if command_requires_provision_config_instance:
        if args.distribution_version:
            cfg.add(config.Scope.applicationOverride, "builder", "distribution.version", args.distribution_version)
        cfg.add(config.Scope.applicationOverride, "builder", "distribution.repository", args.distribution_repository)
        cfg.add(config.Scope.applicationOverride, "builder",
        "provision_config_instance.names", opts.csv_to_list(
            args.provision_config_instance))
        cfg.add(config.Scope.applicationOverride, "builder",
        "provision_config_instance.params", opts.to_dict(
            args.provision_config_instance_params))


def configure_connection_params(arg_parser, args, cfg):
    # Check if multiple hosts are specified using comma separator
    if args.target_hosts and "," in args.target_hosts:
        console.warn(
            "WARNING: Benchmark runs with multiple target hosts should be passed in as a JSON file such as:\n"
            "{\n"
            '  "default": [\n'
            '    {"host": "127.0.0.1", "port": 9200} # Specify nodes for cluster 1\n'
            "  ],\n"
            '  "remote":[\n'
            '    {"host": "10.127.0.3", "port": 9200} # Specify nodes for cluster 2\n'
            "  ]\n"
            "}"
        )
    # Also needed by builder (-> telemetry) - duplicate by module?
    target_hosts = opts.TargetHosts(args.target_hosts)
    cfg.add(config.Scope.applicationOverride, "client", "hosts", target_hosts)
    client_options = opts.ClientOptions(args.client_options, target_hosts=target_hosts)
    cfg.add(config.Scope.applicationOverride, "client", "options", client_options)
    if "timeout" not in client_options.default:
        console.info("You did not provide an explicit timeout in the client options. Assuming default of 10 seconds.")
    if list(target_hosts.all_hosts) != list(client_options.all_client_options):
        arg_parser.error("--target-hosts and --client-options must define the same keys for multi cluster setups.")


def configure_results_publishing_params(args, cfg):
    cfg.add(config.Scope.applicationOverride, "results_publishing", "format", args.results_format)
    cfg.add(config.Scope.applicationOverride, "results_publishing", "values", args.show_in_results)
    cfg.add(config.Scope.applicationOverride, "results_publishing", "output.path", args.results_file)
    cfg.add(config.Scope.applicationOverride, "results_publishing", "numbers.align", args.results_numbers_align)

def prepare_test_executions_dict(args, cfg):
    cfg.add(config.Scope.applicationOverride, "results_publishing", "output.path", args.results_file)
    test_executions_dict = {}
    if args.test_executions:
        for execution in args.test_executions:
            execution = execution.strip()
            if execution:
                test_executions_dict[execution] = None
    return test_executions_dict

def configure_test(arg_parser, args, cfg):
    # As the execute-test command is doing more work than necessary at the moment, we duplicate several parameters
    # in this section that actually belong to dedicated subcommands (like install, start or stop). Over time
    # these duplicated parameters will vanish as we move towards dedicated subcommands and use "execute-test" only
    # to run the actual benchmark (i.e. generating load).
    print_test_execution_id(args)
    if args.effective_start_date:
        cfg.add(config.Scope.applicationOverride, "system", "time.start", args.effective_start_date)
    cfg.add(config.Scope.applicationOverride, "system", "test_execution.id", args.test_execution_id)
    # use the test_execution id implicitly also as the install id.
    cfg.add(config.Scope.applicationOverride, "system", "install.id", args.test_execution_id)
    cfg.add(config.Scope.applicationOverride, "test_execution", "pipeline", args.pipeline)
    cfg.add(config.Scope.applicationOverride, "test_execution", "user.tag", args.user_tag)
    cfg.add(config.Scope.applicationOverride, "worker_coordinator", "profiling", args.enable_worker_coordinator_profiling)
    cfg.add(config.Scope.applicationOverride, "worker_coordinator", "assertions", args.enable_assertions)
    cfg.add(config.Scope.applicationOverride, "worker_coordinator", "on.error", args.on_error)
    cfg.add(
        config.Scope.applicationOverride,
        "worker_coordinator",
        "load_worker_coordinator_hosts",
        opts.csv_to_list(args.load_worker_coordinator_hosts))
    cfg.add(config.Scope.applicationOverride, "workload", "test.mode.enabled", args.test_mode)
    cfg.add(config.Scope.applicationOverride, "workload", "load.test.clients", int(args.load_test_qps))
    if args.redline_test:
        cfg.add(config.Scope.applicationOverride, "workload", "redline.test", int(args.redline_test))
        cfg.add(config.Scope.applicationOverride, "workload", "redline.scale_step", args.redline_scale_step)
        cfg.add(config.Scope.applicationOverride, "workload", "redline.scale_down_pct", args.redline_scaledown_percentage)
        cfg.add(config.Scope.applicationOverride, "workload", "redline.sleep_seconds", args.redline_post_scaledown_sleep)
        cfg.add(config.Scope.applicationOverride, "workload", "redline.max_clients", args.redline_max_clients)
        cfg.add(config.Scope.applicationOverride, "workload", "redline.max_cpu_usage", args.redline_max_cpu_usage)
        cfg.add(config.Scope.applicationOverride, "workload", "redline.cpu_window_seconds", args.redline_cpu_window_seconds)
        cfg.add(config.Scope.applicationOverride, "workload", "redline.cpu_check_interval", args.redline_cpu_check_interval)
    cfg.add(config.Scope.applicationOverride, "workload", "latency.percentiles", args.latency_percentiles)
    cfg.add(config.Scope.applicationOverride, "workload", "throughput.percentiles", args.throughput_percentiles)
    cfg.add(config.Scope.applicationOverride, "workload", "randomization.enabled", args.randomization_enabled)
    cfg.add(config.Scope.applicationOverride, "workload", "randomization.repeat_frequency", args.randomization_repeat_frequency)
    cfg.add(config.Scope.applicationOverride, "workload", "randomization.n", args.randomization_n)
    cfg.add(config.Scope.applicationOverride, "workload", "randomization.alpha", args.randomization_alpha)
    configure_workload_params(arg_parser, args, cfg)
    configure_connection_params(arg_parser, args, cfg)
    configure_telemetry_params(args, cfg)
    configure_builder_params(args, cfg)
    cfg.add(config.Scope.applicationOverride, "builder", "runtime.jdk", args.runtime_jdk)
    cfg.add(config.Scope.applicationOverride, "builder", "source.revision", args.revision)
    cfg.add(config.Scope.applicationOverride, "builder",
    "provision_config_instance.plugins", opts.csv_to_list(
        args.opensearch_plugins))
    cfg.add(config.Scope.applicationOverride, "builder", "plugin.params", opts.to_dict(args.plugin_params))
    cfg.add(config.Scope.applicationOverride, "builder", "preserve.install", convert.to_bool(args.preserve_install))
    cfg.add(config.Scope.applicationOverride, "builder", "skip.rest.api.check", convert.to_bool(args.skip_rest_api_check))

    configure_results_publishing_params(args, cfg)

def print_test_execution_id(args):
    console.info(f"[Test Execution ID]: {args.test_execution_id}")

def dispatch_sub_command(arg_parser, args, cfg):
    sub_command = args.subcommand

    cfg.add(config.Scope.application, "system", "quiet.mode", args.quiet)
    cfg.add(config.Scope.application, "system", "offline.mode", args.offline)

    try:
        if sub_command == "compare":
            configure_results_publishing_params(args, cfg)
            cfg.add(config.Scope.applicationOverride, "results_publishing", "percentiles", args.percentiles)
            results_publisher.compare(cfg, args.baseline, args.contender)
        elif sub_command == "aggregate":
            test_executions_dict = prepare_test_executions_dict(args, cfg)
            aggregator_instance = aggregator.Aggregator(cfg, test_executions_dict, args)
            aggregator_instance.aggregate()
        elif sub_command == "list":
            cfg.add(config.Scope.applicationOverride, "system", "list.config.option", args.configuration)
            cfg.add(config.Scope.applicationOverride, "system", "list.test_executions.max_results", args.limit)
            configure_builder_params(args, cfg, command_requires_provision_config_instance=False)
            configure_workload_params(arg_parser, args, cfg, command_requires_workload=False)
            dispatch_list(cfg)
        elif sub_command == "download":
            cfg.add(config.Scope.applicationOverride, "builder", "target.os", args.target_os)
            cfg.add(config.Scope.applicationOverride, "builder", "target.arch", args.target_arch)
            configure_builder_params(args, cfg)
            builder.download(cfg)
        elif sub_command == "install":
            cfg.add(config.Scope.applicationOverride, "system", "install.id", str(uuid.uuid4()))
            cfg.add(config.Scope.applicationOverride, "builder", "network.host", args.network_host)
            cfg.add(config.Scope.applicationOverride, "builder", "network.http.port", args.http_port)
            cfg.add(config.Scope.applicationOverride, "builder", "source.revision", args.revision)
            cfg.add(config.Scope.applicationOverride, "builder", "build.type", args.build_type)
            cfg.add(config.Scope.applicationOverride, "builder", "runtime.jdk", args.runtime_jdk)
            cfg.add(config.Scope.applicationOverride, "builder", "node.name", args.node_name)
            cfg.add(config.Scope.applicationOverride, "builder", "master.nodes", opts.csv_to_list(args.master_nodes))
            cfg.add(config.Scope.applicationOverride, "builder", "seed.hosts", opts.csv_to_list(args.seed_hosts))
            cfg.add(config.Scope.applicationOverride, "builder",
            "provision_config_instance.plugins", opts.csv_to_list(
                args.opensearch_plugins))
            cfg.add(config.Scope.applicationOverride, "builder", "plugin.params", opts.to_dict(args.plugin_params))
            configure_builder_params(args, cfg)
            builder.install(cfg)
        elif sub_command == "start":
            print_test_execution_id(args)
            cfg.add(config.Scope.applicationOverride, "system", "test_execution.id", args.test_execution_id)
            cfg.add(config.Scope.applicationOverride, "system", "install.id", args.installation_id)
            cfg.add(config.Scope.applicationOverride, "builder", "runtime.jdk", args.runtime_jdk)
            configure_telemetry_params(args, cfg)
            builder.start(cfg)
        elif sub_command == "stop":
            cfg.add(config.Scope.applicationOverride, "builder", "preserve.install", convert.to_bool(args.preserve_install))
            cfg.add(config.Scope.applicationOverride, "system", "install.id", args.installation_id)
            builder.stop(cfg)
        elif sub_command == "execute-test":
            iterations = int(args.test_iterations)
            if iterations > 1:
                test_exes = []
                for _ in range(iterations):
                    try:
                        configure_test(arg_parser, args, cfg)
                        execute_test(cfg, args.kill_running_processes)
                        time.sleep(int(args.sleep_timer))
                        test_exes.append(args.test_execution_id)
                        args.test_execution_id = str(uuid.uuid4())
                    except Exception as e:
                        console.error(f"Error occurred during test execution {_+1}: {str(e)}")
                        if args.cancel_on_error:
                            console.info("Cancelling remaining test executions.")
                            break

                if args.aggregate:
                    args.test_executions = test_exes
                    test_executions_dict = prepare_test_executions_dict(args, cfg)
                    aggregator_instance = aggregator.Aggregator(cfg, test_executions_dict, args)
                    aggregator_instance.aggregate()
            elif args.test_iterations == 1:
                configure_test(arg_parser, args, cfg)
                execute_test(cfg, args.kill_running_processes)
            else:
                console.info("Please enter a valid number of test iterations")
        elif sub_command == "create-workload":
            cfg.add(config.Scope.applicationOverride, "generator", "indices", args.indices)
            cfg.add(config.Scope.applicationOverride, "generator", "number_of_docs", args.number_of_docs)
            cfg.add(config.Scope.applicationOverride, "generator", "output.path", args.output_path)
            cfg.add(config.Scope.applicationOverride, "workload", "workload.name", args.workload)
            cfg.add(config.Scope.applicationOverride, "workload", "custom_queries", args.custom_queries)
            cfg.add(config.Scope.applicationOverride, "generator", "sample_frequency", args.sample_frequency)
            configure_connection_params(arg_parser, args, cfg)

            workload_generator.create_workload(cfg)
        elif sub_command == "info":
            configure_workload_params(arg_parser, args, cfg)
            workload.workload_info(cfg)
        else:
            raise exceptions.SystemSetupError(f"Unknown subcommand [{sub_command}]")
        return True
    except exceptions.BenchmarkError as e:
        logging.getLogger(__name__).exception("Cannot run subcommand [%s].", sub_command)
        msg = str(e.message)
        nesting = 0
        while hasattr(e, "cause") and e.cause:
            nesting += 1
            e = e.cause
            if hasattr(e, "message"):
                msg += "\n%s%s" % ("\t" * nesting, e.message)
            else:
                msg += "\n%s%s" % ("\t" * nesting, str(e))

        console.error("Cannot %s. %s" % (sub_command, msg))
        console.println("")
        print_help_on_errors()
        return False
    except BaseException as e:
        logging.getLogger(__name__).exception("A fatal error occurred while running subcommand [%s].", sub_command)
        console.error("Cannot %s. %s." % (sub_command, e))
        console.println("")
        print_help_on_errors()
        return False


def main():
    check_python_version()
    log.install_default_log_config()
    log.configure_logging()
    logger = logging.getLogger(__name__)
    start = time.time()

    # Early init of console output so we start to show everything consistently.
    console.init(quiet=False)

    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()

    console.init(quiet=args.quiet)
    console.println(BANNER)

    cfg = config.Config(config_name=args.configuration_name)
    if not cfg.config_present():
        cfg.install_default_config()
    cfg.load_config(auto_upgrade=True)
    cfg.add(config.Scope.application, "system", "time.start", datetime.datetime.utcnow())
    # Local config per node
    cfg.add(config.Scope.application, "node", "benchmark.root", paths.benchmark_root())
    cfg.add(config.Scope.application, "node", "benchmark.cwd", os.getcwd())

    logger.info("OS [%s]", str(platform.uname()))
    logger.info("Python [%s]", str(sys.implementation))
    logger.info("OSB version [%s]", version.version())
    logger.debug("Command line arguments: %s", args)
    # Configure networking
    net.init()
    if not args.offline:
        probing_url = cfg.opts("system", "probing.url", default_value="https://github.com", mandatory=False)
        if not net.has_internet_connection(probing_url):
            console.warn("No Internet connection detected. Automatic download of workload data sets etc. is disabled.",
                         logger=logger)
            cfg.add(config.Scope.applicationOverride, "system", "offline.mode", True)
        else:
            logger.info("Detected a working Internet connection.")

    success = dispatch_sub_command(arg_parser, args, cfg)

    end = time.time()
    if success:
        console.println("")
        console.info("SUCCESS (took %d seconds)" % (end - start), overline="-", underline="-")
    else:
        console.println("")
        console.info("FAILURE (took %d seconds)" % (end - start), overline="-", underline="-")
        sys.exit(64)


if __name__ == "__main__":
    main()
