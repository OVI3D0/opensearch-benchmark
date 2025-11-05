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
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import datetime
import itertools
import logging
import queue

import time

from osbenchmark import actor, profiler
from osbenchmark.utils import convert

class MetricsProcessor(actor.BenchmarkActor):
    WAKEUP_INTERVAL = 1
    FLUSH_INTERVAL_SECONDS = 60  # Flush metrics to OpenSearch every 60 seconds

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.flush_counter = 0  # Counter to track when to flush

    def receiveMsg_StartMetricsProcessor(self, msg, sender) -> None:
        from osbenchmark import metrics
        from osbenchmark.worker_coordinator.worker_coordinator import load_local_config

        self.logger.info("MetricsProcessor starting with downsample_factor=%d", msg.downsample_factor)
        self.raw_samples = []
        self.raw_profile_samples = []
        self.most_recent_sample_per_client = {}

        # Create our own metrics_store from config
        config = load_local_config(msg.config)
        self.metrics_store = metrics.metrics_store(
            cfg=config,
            workload=msg.workload_name,
            test_procedure=msg.test_procedure_name,
            read_only=False
        )

        self.workload_meta_data = msg.workload_meta_data
        self.test_procedure_meta_data = msg.test_procedure_meta_data

        # Create the sample post-processor
        self.sample_post_processor = DefaultSamplePostprocessor(
            self.metrics_store,
            msg.downsample_factor,
            msg.workload_meta_data,
            msg.test_procedure_meta_data
        )
        self.profile_metrics_post_processor = None

        self.wakeupAfter(datetime.timedelta(seconds=MetricsProcessor.WAKEUP_INTERVAL))

    def receiveMsg_WakeupMessage(self, msg, sender) -> None:
        self.process_metrics()
        self.wakeupAfter(datetime.timedelta(seconds=MetricsProcessor.WAKEUP_INTERVAL))

    def receiveMsg_UpdateSamples(self, msg, sender) -> None:
        samples = msg.samples
        self.logger.info(f"UPDATE_SAMPLES CALLED: len(samples)={len(samples)}, profiler.enabled={profiler._profiler.enabled}")
        try:
            with profiler.ProfileContext("update_samples"):
                if len(samples) > 0:
                    self.raw_samples += samples
                    # We need to check all samples, they will be from different clients
                    for s in samples:
                        self.most_recent_sample_per_client[s.client_id] = s
            self.logger.info(f"UPDATE_SAMPLES completed, current stats count: {len(profiler.get_stats())}")
        except Exception as e:
            self.logger.error(f"UPDATE_SAMPLES ProfileContext error: {e}", exc_info=True)

    def receiveMsg_ActorExitRequest(self, msg, sender):
        self.logger.info("Metrics Processor received ActorExitRequest. Shutting down...")
        # Flush any remaining metrics before closing
        if hasattr(self, 'metrics_store') and self.metrics_store:
            try:
                self.logger.info("Final flush of metrics store before shutdown...")
                self.metrics_store.flush(refresh=True)
                self.logger.info("MetricsProcessor flushed metrics store successfully")
                self.metrics_store.close()
                self.logger.info("MetricsProcessor closed metrics store successfully")
            except Exception as e:
                self.logger.warning("Error flushing/closing metrics store: %s", e)
    
    def update_samples(self, samples):
        self.logger.info(f"UPDATE_SAMPLES CALLED: len(samples)={len(samples)}, profiler.enabled={profiler._profiler.enabled}")
        try:
            with profiler.ProfileContext("update_samples"):
                if len(samples) > 0:
                    self.raw_samples += samples
                    # We need to check all samples, they will be from different clients
                    for s in samples:
                        self.most_recent_sample_per_client[s.client_id] = s
            self.logger.info(f"UPDATE_SAMPLES completed, current stats count: {len(profiler.get_stats())}")
        except Exception as e:
            self.logger.error(f"UPDATE_SAMPLES ProfileContext error: {e}", exc_info=True)

    def process_metrics(self):
        self.logger.info(f"POST_PROCESS_SAMPLES CALLED: len(raw_samples)={len(self.raw_samples)}, profiler.enabled={profiler._profiler.enabled}")
        with profiler.ProfileContext("post_process_samples"):
            # we do *not* do this here to avoid concurrent updates (actors are single-threaded) but rather to make it clear that we use
            # only a snapshot and that new data will go to a new sample set.
            raw_samples = self.raw_samples
            self.raw_samples = []
            self.sample_post_processor(raw_samples)
            profile_samples = self.raw_profile_samples
            self.raw_profile_samples = []
            if len(profile_samples) > 0:
                if self.profile_metrics_post_processor is None:
                    self.profile_metrics_post_processor = ProfileMetricsSamplePostprocessor(self.metrics_store,
                                                                                        self.workload_meta_data,
                                                                                        self.test_procedure_meta_data)
                self.profile_metrics_post_processor(profile_samples)

            # Periodically flush processed metrics to OpenSearch to prevent memory accumulation
            self.flush_counter += MetricsProcessor.WAKEUP_INTERVAL
            if self.flush_counter >= MetricsProcessor.FLUSH_INTERVAL_SECONDS:
                self.logger.info(f"Flushing metrics store (periodic flush every {MetricsProcessor.FLUSH_INTERVAL_SECONDS}s)")
                self.metrics_store.flush(refresh=False)  # Don't refresh to save time
                self.flush_counter = 0

class SamplePostprocessor():
    """
    Parent class used to process samples into the metrics store
    """
    def __init__(self, metrics_store, workload_meta_data, test_procedure_meta_data):
        self.logger = logging.getLogger(__name__)
        self.metrics_store = metrics_store
        self.workload_meta_data = workload_meta_data
        self.test_procedure_meta_data = test_procedure_meta_data

    def merge(self, *args):
        result = {}
        for arg in args:
            if arg is not None:
                result.update(arg)
        return result


class DefaultSamplePostprocessor(SamplePostprocessor):
    """
    Processes operational and correctness metric samples by merging and adding to the metrics store
    """
    def __init__(self, metrics_store, downsample_factor, workload_meta_data, test_procedure_meta_data):
        super().__init__(metrics_store, workload_meta_data, test_procedure_meta_data)
        self.throughput_calculator = ThroughputCalculator()
        self.downsample_factor = downsample_factor

    def __call__(self, raw_samples):
        with profiler.ProfileContext("sample_postprocessor"):
            if len(raw_samples) == 0:
                return
            total_start = time.perf_counter()
            start = total_start
            final_sample_count = 0
            for idx, sample in enumerate(raw_samples):
                self.logger.debug(
                    "All sample meta data: [%s],[%s],[%s],[%s],[%s]",
                    self.workload_meta_data,
                    self.test_procedure_meta_data,
                    sample.operation_meta_data,
                    sample.task.meta_data,
                    sample.request_meta_data,
                )

                # if request_meta_data exists then it will have {"success": true/false} as a parameter.
                if sample.request_meta_data and len(sample.request_meta_data) > 1:
                    self.logger.debug("Found: %s", sample.request_meta_data)

                    recall_metric_names = ["recall@k", "recall@1"]

                    for recall_metric_name in recall_metric_names:
                        if recall_metric_name in sample.request_meta_data:
                            meta_data = self.merge(
                                self.workload_meta_data,
                                self.test_procedure_meta_data,
                                sample.operation_meta_data,
                                sample.task.meta_data,
                                sample.request_meta_data,
                            )

                            self.metrics_store.put_value_cluster_level(
                                name=recall_metric_name,
                                value=sample.request_meta_data[recall_metric_name],
                                unit="",
                                task=sample.task.name,
                                operation=sample.operation_name,
                                operation_type=sample.operation_type,
                                sample_type=sample.sample_type,
                                absolute_time=sample.absolute_time,
                                relative_time=sample.relative_time,
                                meta_data=meta_data,
                            )

                if idx % self.downsample_factor == 0:
                    final_sample_count += 1
                    meta_data = self.merge(
                        self.workload_meta_data,
                        self.test_procedure_meta_data,
                        sample.operation_meta_data,
                        sample.task.meta_data,
                        sample.request_meta_data)

                    self.metrics_store.put_value_cluster_level(name="latency", value=convert.seconds_to_ms(sample.latency),
                                                               unit="ms", task=sample.task.name,
                                                               operation=sample.operation_name, operation_type=sample.operation_type,
                                                               sample_type=sample.sample_type, absolute_time=sample.absolute_time,
                                                               relative_time=sample.relative_time, meta_data=meta_data)

                    self.metrics_store.put_value_cluster_level(name="service_time", value=convert.seconds_to_ms(sample.service_time),
                                                               unit="ms", task=sample.task.name,
                                                               operation=sample.operation_name, operation_type=sample.operation_type,
                                                               sample_type=sample.sample_type, absolute_time=sample.absolute_time,
                                                               relative_time=sample.relative_time, meta_data=meta_data)

                    self.metrics_store.put_value_cluster_level(name="client_processing_time",
                                                               value=convert.seconds_to_ms(sample.client_processing_time),
                                                               unit="ms", task=sample.task.name,
                                                               operation=sample.operation_name, operation_type=sample.operation_type,
                                                               sample_type=sample.sample_type, absolute_time=sample.absolute_time,
                                                               relative_time=sample.relative_time, meta_data=meta_data)

                    self.metrics_store.put_value_cluster_level(name="processing_time", value=convert.seconds_to_ms(sample.processing_time),
                                                               unit="ms", task=sample.task.name,
                                                               operation=sample.operation_name, operation_type=sample.operation_type,
                                                               sample_type=sample.sample_type, absolute_time=sample.absolute_time,
                                                               relative_time=sample.relative_time, meta_data=meta_data)

                    for timing in sample.dependent_timings:
                        self.metrics_store.put_value_cluster_level(name="service_time", value=convert.seconds_to_ms(timing.service_time),
                                                                   unit="ms", task=timing.task.name,
                                                                   operation=timing.operation_name, operation_type=timing.operation_type,
                                                                   sample_type=timing.sample_type, absolute_time=timing.absolute_time,
                                                                   relative_time=timing.relative_time, meta_data=meta_data)

            end = time.perf_counter()
            self.logger.debug("Storing latency and service time took [%f] seconds.", (end - start))
            start = end
            aggregates = self.throughput_calculator.calculate(raw_samples)
            end = time.perf_counter()
            self.logger.debug("Calculating throughput took [%f] seconds.", (end - start))
            start = end
            for task, samples in aggregates.items():
                meta_data = self.merge(
                    self.workload_meta_data,
                    self.test_procedure_meta_data,
                    task.operation.meta_data,
                    task.meta_data
                )
                for absolute_time, relative_time, sample_type, throughput, throughput_unit in samples:
                    self.metrics_store.put_value_cluster_level(name="throughput", value=throughput, unit=throughput_unit, task=task.name,
                                                               operation=task.operation.name, operation_type=task.operation.type,
                                                               sample_type=sample_type, absolute_time=absolute_time,
                                                               relative_time=relative_time, meta_data=meta_data)
            end = time.perf_counter()
            self.logger.debug("Storing throughput took [%f] seconds.", (end - start))
            start = end
            # this will be a noop for the in-memory metrics store. If we use an ES metrics store however, this will ensure that we already send
            # the data and also clear the in-memory buffer. This allows users to see data already while running the benchmark. In cases where
            # it does not matter (i.e. in-memory) we will still defer this step until the end.
            #
            # Don't force refresh here in the interest of short processing times. We don't need to query immediately afterwards so there is
            # no need for frequent refreshes.
            self.metrics_store.flush(refresh=False)
            end = time.perf_counter()
            self.logger.debug("Flushing the metrics store took [%f] seconds.", (end - start))
            self.logger.debug("Postprocessing [%d] raw samples (downsampled to [%d] samples) took [%f] seconds in total.",
                              len(raw_samples), final_sample_count, (end - total_start))


class ProfileMetricsSamplePostprocessor(SamplePostprocessor):
    """
    Processes profile metric samples by merging and adding to the metrics store
    """

    def __call__(self, raw_samples):
        if len(raw_samples) == 0:
            return
        total_start = time.perf_counter()
        start = total_start
        final_sample_count = 0
        for sample in raw_samples:
            final_sample_count += 1
            self.logger.debug(
                "All sample meta data: [%s],[%s],[%s],[%s],[%s]",
                self.workload_meta_data,
                self.test_procedure_meta_data,
                sample.operation_meta_data,
                sample.task.meta_data,
                sample.request_meta_data,
            )

            # if request_meta_data exists then it will have {"success": true/false} as a parameter.
            if sample.request_meta_data and len(sample.request_meta_data) > 1:
                self.logger.debug("Found: %s", sample.request_meta_data)

                if "profile-metrics" in sample.request_meta_data:
                    for metric_name, metric_value in sample.request_meta_data["profile-metrics"].items():
                        meta_data = self.merge(
                            self.workload_meta_data,
                            self.test_procedure_meta_data,
                            sample.operation_meta_data,
                            sample.task.meta_data,
                            sample.request_meta_data,
                        )

                        self.metrics_store.put_value_cluster_level(
                            name=metric_name,
                            value=metric_value,
                            unit="",
                            task=sample.task.name,
                            operation=sample.operation_name,
                            operation_type=sample.operation_type,
                            sample_type=sample.sample_type,
                            absolute_time=sample.absolute_time,
                            relative_time=sample.relative_time,
                            meta_data=meta_data,
                        )

        start = time.perf_counter()
        # this will be a noop for the in-memory metrics store. If we use an ES metrics store however, this will ensure that we already send
        # the data and also clear the in-memory buffer. This allows users to see data already while running the benchmark. In cases where
        # it does not matter (i.e. in-memory) we will still defer this step until the end.
        #
        # Don't force refresh here in the interest of short processing times. We don't need to query immediately afterwards so there is
        # no need for frequent refreshes.
        self.metrics_store.flush(refresh=False)
        end = time.perf_counter()
        self.logger.debug("Flushing the metrics store took [%f] seconds.", (end - start))
        self.logger.debug("Postprocessing [%d] raw samples (downsampled to [%d] samples) took [%f] seconds in total.",
                          len(raw_samples), final_sample_count, (end - total_start))


class Sampler:
    """
    Encapsulates management of gathered samples.
    """

    def __init__(self, start_timestamp, buffer_size=16384):
        self.start_timestamp = start_timestamp
        self.q = queue.Queue(maxsize=buffer_size)
        self.logger = logging.getLogger(__name__)

    @property
    def samples(self):
        samples = []
        try:
            while True:
                samples.append(self.q.get_nowait())
        except queue.Empty:
            pass
        return samples

class DefaultSampler(Sampler):
    """
    Encapsulates management of gathered default samples (operational and correctness metrics).
    """

    def add(self, task, client_id, sample_type, meta_data, absolute_time, request_start, latency, service_time,
            client_processing_time, processing_time, throughput, ops, ops_unit, time_period, percent_completed,
            dependent_timing=None):
        try:
            self.q.put_nowait(
                DefaultSample(client_id, absolute_time, request_start, self.start_timestamp, task, sample_type, meta_data,
                       latency, service_time, client_processing_time, processing_time, throughput, ops, ops_unit, time_period,
                       percent_completed, dependent_timing))
        except queue.Full:
            self.logger.warning("Dropping sample for [%s] due to a full sampling queue.", task.operation.name)

class ProfileMetricsSampler(Sampler):
    """
    Encapsulates management of gathered profile metrics samples.
    """

    def add(self, task, client_id, sample_type, meta_data, absolute_time, request_start, time_period, percent_completed,
            dependent_timing=None):
        try:
            self.q.put_nowait(
                ProfileMetricsSample(client_id, absolute_time, request_start, self.start_timestamp, task, sample_type, meta_data,
                       time_period, percent_completed, dependent_timing))
        except queue.Full:
            self.logger.warning("Dropping sample for [%s] due to a full sampling queue.", task.operation.name)


class Sample:
    """
    Basic information used by metrics store to keep track of samples
    """
    __slots__ = ('client_id', 'absolute_time', 'request_start', 'task_start', 'task',
                 'sample_type', 'request_meta_data', 'time_period', '_dependent_timing',
                 'percent_completed')

    def __init__(self, client_id, absolute_time, request_start, task_start, task, sample_type, request_meta_data,
                time_period, percent_completed, dependent_timing=None):
        self.client_id = client_id
        self.absolute_time = absolute_time
        self.request_start = request_start
        self.task_start = task_start
        self.task = task
        self.sample_type = sample_type
        self.request_meta_data = request_meta_data
        self.time_period = time_period
        self._dependent_timing = dependent_timing
        # may be None for eternal tasks!
        self.percent_completed = percent_completed

    @property
    def operation_name(self):
        return self.task.operation.name

    @property
    def operation_type(self):
        return self.task.operation.type

    @property
    def operation_meta_data(self):
        return self.task.operation.meta_data

    @property
    def relative_time(self):
        return self.request_start - self.task_start

    def __repr__(self, *args, **kwargs):
        return f"[{self.absolute_time}; {self.relative_time}] [client [{self.client_id}]] [{self.task}] " \
               f"[{self.sample_type}]"

class DefaultSample(Sample):
    """
    Stores the operational and correctness metrics to later put into the metrics store
    """
    __slots__ = ('latency', 'service_time', 'client_processing_time', 'processing_time',
                 'throughput', 'total_ops', 'total_ops_unit')

    def __init__(self, client_id, absolute_time, request_start, task_start, task, sample_type, request_meta_data, latency,
                 service_time, client_processing_time, processing_time, throughput, total_ops, total_ops_unit, time_period,
                 percent_completed, dependent_timing=None):
        super().__init__(client_id, absolute_time, request_start, task_start, task, sample_type, request_meta_data, time_period, percent_completed, dependent_timing)
        self.latency = latency
        self.service_time = service_time
        self.client_processing_time = client_processing_time
        self.processing_time = processing_time
        self.throughput = throughput
        self.total_ops = total_ops
        self.total_ops_unit = total_ops_unit

    @property
    def dependent_timings(self):
        if self._dependent_timing:
            for t in self._dependent_timing:
                yield DefaultSample(self.client_id, t["absolute_time"], t["request_start"], self.task_start, self.task,
                             self.sample_type, self.request_meta_data, 0, t["service_time"], 0, 0, 0, self.total_ops,
                             self.total_ops_unit, self.time_period, self.percent_completed, None)

    def __repr__(self, *args, **kwargs):
        return f"[{self.absolute_time}; {self.relative_time}] [client [{self.client_id}]] [{self.task}] " \
               f"[{self.sample_type}]: [{self.latency}s] request latency, [{self.service_time}s] service time, " \
               f"[{self.total_ops} {self.total_ops_unit}]"

class ProfileMetricsSample(Sample):
    """
    Stores the profile metrics to later put into the metrics store
    """
    __slots__ = ()  # No additional attributes beyond Sample

    @property
    def dependent_timings(self):
        if self._dependent_timing:
            for t in self._dependent_timing:
                yield ProfileMetricsSample(self.client_id, t["absolute_time"], t["request_start"], self.task_start, self.task,
                             self.sample_type, self.request_meta_data, self.time_period, self.percent_completed, None)


class ThroughputCalculator:
    class TaskStats:
        """
        Stores per task numbers needed for throughput calculation in between multiple calculations.
        """
        def __init__(self, bucket_interval, sample_type, start_time):
            self.unprocessed = []
            self.total_count = 0
            self.interval = 0
            self.bucket_interval = bucket_interval
            # the first bucket is complete after one bucket interval is over
            self.bucket = bucket_interval
            self.sample_type = sample_type
            self.has_samples_in_sample_type = False
            # start relative to the beginning of our (calculation) time slice.
            self.start_time = start_time

        @property
        def throughput(self):
            return self.total_count / self.interval

        def maybe_update_sample_type(self, current_sample_type):
            if self.sample_type < current_sample_type:
                self.sample_type = current_sample_type
                self.has_samples_in_sample_type = False

        def update_interval(self, absolute_sample_time):
            self.interval = max(absolute_sample_time - self.start_time, self.interval)

        def can_calculate_throughput(self):
            return self.interval > 0 and self.interval >= self.bucket

        def can_add_final_throughput_sample(self):
            return self.interval > 0 and not self.has_samples_in_sample_type

        def finish_bucket(self, new_total):
            self.unprocessed = []
            self.total_count = new_total
            self.has_samples_in_sample_type = True
            self.bucket = int(self.interval) + self.bucket_interval

    def __init__(self):
        self.task_stats = {}

    def calculate(self, samples, bucket_interval_secs=1):
        """
        Calculates global throughput based on samples gathered from multiple load generators.

        :param samples: A list containing all samples from all load generators.
        :param bucket_interval_secs: The bucket interval for aggregations.
        :return: A global view of throughput samples.
        """
        with profiler.ProfileContext("throughput_calculation"):
            samples_per_task = {}
            # first we group all samples by task (operation).
            for sample in samples:
                k = sample.task
                if k not in samples_per_task:
                    samples_per_task[k] = []
                samples_per_task[k].append(sample)

            global_throughput = {}
            # with open("raw_samples_new.csv", "a") as sample_log:
            # print("client_id,absolute_time,relative_time,operation,sample_type,total_ops,time_period", file=sample_log)
            for k, v in samples_per_task.items():
                task = k
                if task not in global_throughput:
                    global_throughput[task] = []
                # sort all samples by time
                if task in self.task_stats:
                    samples = itertools.chain(v, self.task_stats[task].unprocessed)
                else:
                    samples = v
                current_samples = sorted(samples, key=lambda s: s.absolute_time)

                # Calculate throughput based on service time if the runner does not provide one, otherwise use it as is and
                # only transform the values into the expected structure.
                first_sample = current_samples[0]
                if first_sample.throughput is None:
                    task_throughput = self.calculate_task_throughput(task, current_samples, bucket_interval_secs)
                else:
                    task_throughput = self.map_task_throughput(current_samples)
                global_throughput[task].extend(task_throughput)

            return global_throughput

    def calculate_task_throughput(self, task, current_samples, bucket_interval_secs):
        task_throughput = []

        if task not in self.task_stats:
            first_sample = current_samples[0]
            self.task_stats[task] = ThroughputCalculator.TaskStats(bucket_interval=bucket_interval_secs,
                                                                   sample_type=first_sample.sample_type,
                                                                   start_time=first_sample.absolute_time - first_sample.time_period)
        current = self.task_stats[task]
        count = current.total_count
        last_sample = None
        for sample in current_samples:
            last_sample = sample
            # print("%d,%f,%f,%s,%s,%d,%f" %
            #       (sample.client_id, sample.absolute_time, sample.relative_time, sample.operation, sample.sample_type,
            #        sample.total_ops, sample.time_period), file=sample_log)

            # once we have seen a new sample type, we stick to it.
            current.maybe_update_sample_type(sample.sample_type)

            # we need to store the total count separately and cannot update `current.total_count` immediately here
            # because we would count all raw samples in `unprocessed` twice. Hence, we'll only update
            # `current.total_count` when we have calculated a new throughput sample.
            count += sample.total_ops
            current.update_interval(sample.absolute_time)

            if current.can_calculate_throughput():
                current.finish_bucket(count)
                task_throughput.append((sample.absolute_time,
                                        sample.relative_time,
                                        current.sample_type,
                                        current.throughput,
                                        # we calculate throughput per second
                                        f"{sample.total_ops_unit}/s"))
            else:
                current.unprocessed.append(sample)

        # also include the last sample if we don't have one for the current sample type, even if it is below the bucket
        # interval (mainly needed to ensure we show throughput data in test mode)
        if last_sample is not None and current.can_add_final_throughput_sample():
            current.finish_bucket(count)
            task_throughput.append((last_sample.absolute_time,
                                    last_sample.relative_time,
                                    current.sample_type,
                                    current.throughput,
                                    f"{last_sample.total_ops_unit}/s"))

        return task_throughput

    def map_task_throughput(self, current_samples):
        throughput = []
        for sample in current_samples:
            throughput.append((sample.absolute_time,
                               sample.relative_time,
                               sample.sample_type,
                               sample.throughput,
                               f"{sample.total_ops_unit}/s"))
        return throughput
