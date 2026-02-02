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

import datetime
import itertools
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Any

import thespian.actors

from osbenchmark import actor
from osbenchmark.utils import convert


##################################
#
# Messages for MetricsActor
#
##################################

@dataclass
class StartMetricsActor:
    """Initialize the MetricsActor with configuration."""
    metrics_store: Any
    downsample_factor: int
    workload_meta_data: dict
    test_procedure_meta_data: dict
    neighbors_dataset: Optional[List] = None  # For recall calculation
    k: int = 100  # For recall@k


@dataclass
class UpdateRawSamples:
    """Send raw samples from a worker to the MetricsActor."""
    worker_id: int
    samples: List[Any]
    profile_samples: List[Any] = field(default_factory=list)


@dataclass
class FlushMetrics:
    """Request the MetricsActor to flush all pending samples."""
    pass


@dataclass
class GetMetricsStats:
    """Request current processing statistics."""
    pass


@dataclass
class MetricsStats:
    """Response with current processing statistics."""
    total_samples_received: int
    total_samples_processed: int
    samples_pending: int
    processing_rate: float  # samples/second


@dataclass
class StopMetricsActor:
    """Signal the MetricsActor to stop processing and flush remaining samples."""
    pass


class MetricsActor(actor.BenchmarkActor):
    """
    Thespian actor that continuously processes metrics samples from workers.

    Unlike the WorkerCoordinator which processes samples every 30 seconds,
    this actor processes samples as fast as possible (every 100ms by default).

    Architecture:
        Workers ──► UpdateRawSamples ──► MetricsActor ──► MetricsStore
                                              │
                                              ├── Queue (bounded, ~100k samples)
                                              ├── Continuous processing loop
                                              └── Recall calculation (if enabled)

    Benefits:
    - Offloads metrics processing from workers (faster hot path)
    - Real-time metrics visibility
    - Bounded memory usage via queue limits
    - Deferred recall calculation (doesn't block search)
    """

    # Process samples every 100ms (vs 30s in WorkerCoordinator)
    PROCESSING_INTERVAL_SECONDS = 0.1
    # Maximum samples to hold in queue before dropping
    MAX_QUEUE_SIZE = 100000

    def __init__(self):
        super().__init__()
        self.metrics_store = None
        self.sample_postprocessor = None
        self.profile_postprocessor = None
        self.metrics_processor = None

        # Sample queue - thread-safe for receiving from multiple workers
        self.sample_queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self.profile_queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)

        # Statistics
        self.total_samples_received = 0
        self.total_samples_processed = 0
        self.processing_start_time = None
        self.is_running = False

    def receiveMsg_StartMetricsActor(self, msg, sender):
        """Initialize the actor with metrics configuration."""
        self.logger.info("MetricsActor starting with downsample_factor=%d", msg.downsample_factor)

        self.metrics_store = msg.metrics_store

        # Create postprocessors (reuse existing classes)
        self.sample_postprocessor = DefaultSamplePostprocessor(
            msg.metrics_store,
            msg.downsample_factor,
            msg.workload_meta_data,
            msg.test_procedure_meta_data
        )
        self.profile_postprocessor = ProfileMetricsSamplePostprocessor(
            msg.metrics_store,
            msg.workload_meta_data,
            msg.test_procedure_meta_data
        )

        # Create metrics processor for recall calculation
        self.metrics_processor = MetricsProcessor(
            neighbors_dataset=msg.neighbors_dataset,
            k=msg.k
        )

        self.processing_start_time = time.perf_counter()
        self.is_running = True

        # Start the processing loop
        self.wakeupAfter(datetime.timedelta(seconds=self.PROCESSING_INTERVAL_SECONDS))

        self.logger.info("MetricsActor initialized and processing loop started")

    def receiveMsg_UpdateRawSamples(self, msg, sender):
        """Receive samples from a worker and queue them for processing."""
        samples_added = 0
        samples_dropped = 0

        # Queue default samples
        for sample in msg.samples:
            try:
                self.sample_queue.put_nowait(sample)
                samples_added += 1
            except queue.Full:
                samples_dropped += 1

        # Queue profile samples
        for sample in msg.profile_samples:
            try:
                self.profile_queue.put_nowait(sample)
            except queue.Full:
                pass  # Profile samples are less critical

        self.total_samples_received += samples_added

        if samples_dropped > 0:
            self.logger.warning(
                "MetricsActor dropped %d samples from worker %d (queue full)",
                samples_dropped, msg.worker_id
            )

    def receiveMsg_WakeupMessage(self, msg, sender):
        """Process pending samples on each wakeup."""
        if not self.is_running:
            return

        self._process_pending_samples()

        # Schedule next processing cycle
        self.wakeupAfter(datetime.timedelta(seconds=self.PROCESSING_INTERVAL_SECONDS))

    def receiveMsg_FlushMetrics(self, msg, sender):
        """Flush all pending samples immediately."""
        self.logger.info("MetricsActor flushing all pending samples")
        self._process_pending_samples(flush_all=True)
        if self.metrics_store:
            self.metrics_store.flush(refresh=True)
        self.logger.info("MetricsActor flush complete")

    def receiveMsg_GetMetricsStats(self, msg, sender):
        """Return current processing statistics."""
        elapsed = time.perf_counter() - self.processing_start_time if self.processing_start_time else 1
        rate = self.total_samples_processed / elapsed if elapsed > 0 else 0

        stats = MetricsStats(
            total_samples_received=self.total_samples_received,
            total_samples_processed=self.total_samples_processed,
            samples_pending=self.sample_queue.qsize(),
            processing_rate=rate
        )
        self.send(sender, stats)

    def receiveMsg_StopMetricsActor(self, msg, sender):
        """Stop processing and flush remaining samples."""
        self.logger.info("MetricsActor stopping, processing remaining samples...")
        self.is_running = False
        self._process_pending_samples(flush_all=True)
        if self.metrics_store:
            self.metrics_store.flush(refresh=True)
        self.logger.info(
            "MetricsActor stopped. Total: received=%d, processed=%d",
            self.total_samples_received, self.total_samples_processed
        )

    def receiveMsg_ActorExitRequest(self, msg, sender):
        """Handle actor exit."""
        self.is_running = False
        self.logger.info("MetricsActor received exit request")

    def _process_pending_samples(self, flush_all=False):
        """
        Process samples from the queue.

        Args:
            flush_all: If True, process ALL pending samples. Otherwise, process
                      a batch to maintain responsiveness.
        """
        # Collect samples from queue
        samples = []
        profile_samples = []

        # Batch size - process up to this many samples per cycle
        # Unless flush_all is True, then process everything
        batch_size = self.MAX_QUEUE_SIZE if flush_all else 10000

        # Drain sample queue
        count = 0
        while count < batch_size:
            try:
                sample = self.sample_queue.get_nowait()
                samples.append(sample)
                count += 1
            except queue.Empty:
                break

        # Drain profile queue
        count = 0
        while count < batch_size:
            try:
                sample = self.profile_queue.get_nowait()
                profile_samples.append(sample)
                count += 1
            except queue.Empty:
                break

        # Process samples using existing postprocessors
        if samples:
            start = time.perf_counter()
            try:
                # Calculate recall for samples with candidate_ids (deferred recall)
                self._calculate_deferred_recall(samples)

                # Use existing postprocessor
                self.sample_postprocessor(samples)
                self.total_samples_processed += len(samples)

                elapsed = time.perf_counter() - start
                if elapsed > 0.5:  # Log if processing takes too long
                    self.logger.debug(
                        "Processed %d samples in %.3fs (%.0f samples/s)",
                        len(samples), elapsed, len(samples) / elapsed
                    )
            except Exception as e:
                self.logger.error("Error processing samples: %s", e, exc_info=True)

        if profile_samples:
            try:
                self.profile_postprocessor(profile_samples)
            except Exception as e:
                self.logger.error("Error processing profile samples: %s", e, exc_info=True)

    def _calculate_deferred_recall(self, samples):
        """
        Calculate recall for samples that have candidate_ids but no recall values.

        This is the deferred recall calculation - workers send candidate_ids
        and query_index, and we calculate recall here instead of in the hot path.
        """
        if self.metrics_processor is None or self.metrics_processor.neighbors_dataset is None:
            return

        for sample in samples:
            meta = getattr(sample, 'request_meta_data', None)
            if meta is None:
                continue

            # Check if this sample needs deferred recall calculation
            if 'candidate_ids' in meta and 'query_index' in meta:
                if 'recall@k' not in meta:  # Don't recalculate if already present
                    recall = self.metrics_processor.calculate_recall(
                        meta['candidate_ids'],
                        meta['query_index']
                    )
                    if recall:
                        meta['recall@k'] = recall['recall@k']
                        meta['recall@1'] = recall['recall@1']

class MetricsProcessor:
    """
    General-purpose metrics processor for OpenSearch Benchmark.

    Processes raw query samples and calculates all OSB metrics including:
    - Throughput (ops/s)
    - Latency percentiles (p50, p90, p99)
    - Service time percentiles
    - Vector search recall (recall@k, recall@1)
    - Error rates

    Can be used in two modes:
    1. Batch mode: Process all samples after benchmark completion
    2. Streaming mode: Process samples as they arrive (for real-time metrics)
    """

    def __init__(self, neighbors_dataset=None, k=100):
        """
        Args:
            neighbors_dataset: Ground truth neighbors for recall calculation.
                              Format: neighbors_dataset[query_idx] = [id1, id2, ...]
            k: Number of neighbors to consider for recall@k calculation
        """
        self.logger = logging.getLogger(__name__)
        self.neighbors_dataset = neighbors_dataset
        self.k = k

    def calculate_recall(self, candidate_ids, query_idx, top_k=None):
        """
        Calculate recall@k and recall@1 for a single query.

        Args:
            candidate_ids: List of IDs returned by the search
            query_idx: Index into the neighbors_dataset for ground truth
            top_k: Number of neighbors to check (defaults to self.k)

        Returns:
            dict with recall@k and recall@1 values, or None if no ground truth
        """
        if self.neighbors_dataset is None or query_idx is None:
            return None

        if query_idx >= len(self.neighbors_dataset):
            self.logger.warning("Query index %d out of bounds for neighbors dataset (size %d)",
                               query_idx, len(self.neighbors_dataset))
            return None

        k = top_k if top_k is not None else self.k
        neighbors = self.neighbors_dataset[query_idx]

        recall_k = self._calculate_topk_recall(candidate_ids, neighbors, k)
        recall_1 = self._calculate_topk_recall(candidate_ids, neighbors, 1)

        return {
            "recall@k": recall_k,
            "recall@1": recall_1
        }

    def _calculate_topk_recall(self, predictions, neighbors, top_k):
        """
        Calculate recall by comparing top_k neighbors with predictions.

        recall = matched neighbors / total ground truth neighbors

        Args:
            predictions: List of IDs returned by OpenSearch
            neighbors: List of ground truth neighbor IDs
            top_k: Number of top results to check

        Returns:
            Recall value between 0 and 1
        """
        if neighbors is None or len(neighbors) == 0:
            self.logger.debug("No neighbors provided for recall calculation")
            return 0.0

        min_num_of_results = min(top_k, len(neighbors))

        # Handle -1 sentinel values in ground truth (indicates no more valid neighbors)
        truth_set = list(neighbors[:min_num_of_results])
        last_neighbor_idx = self._find_last_valid_neighbor(truth_set)
        if last_neighbor_idx < len(truth_set):
            truth_set = truth_set[:last_neighbor_idx]
            if not truth_set:
                self.logger.debug("No true neighbors after filtering -1s, returning recall = 1.0")
                return 1.0

        # Convert to strings for comparison (IDs may be int or str)
        truth_set_str = set(str(n) for n in truth_set)

        correct = 0.0
        for j in range(min(min_num_of_results, len(predictions))):
            if str(predictions[j]) in truth_set_str:
                correct += 1.0

        return correct / len(truth_set) if truth_set else 0.0

    def _find_last_valid_neighbor(self, neighbors):
        """Find index of first -1 in neighbors list (binary search)."""
        if not neighbors:
            return 0

        # Check if last element is -1
        if int(neighbors[-1]) != -1:
            return len(neighbors)

        # Binary search for first -1
        low, high = 0, len(neighbors) - 1
        while low < high:
            mid = (low + high) // 2
            if int(neighbors[mid]) == -1:
                high = mid
            else:
                low = mid + 1
        return low

    def _calculate_radial_recall(self, predictions, neighbors, top_1_only=False):
        """
        Calculate recall for radial (max_distance/min_score) searches.

        Args:
            predictions: List of IDs returned by OpenSearch
            neighbors: List of ground truth neighbor IDs (may contain -1 sentinel)
            top_1_only: If True, only check recall@1

        Returns:
            Recall value between 0 and 1
        """
        if neighbors is None:
            return 1.0

        # Find actual neighbors (stop at -1)
        try:
            n = list(neighbors).index('-1')
            truth_set = list(neighbors[:n])
        except ValueError:
            truth_set = list(neighbors)

        if len(truth_set) == 0:
            return 1.0

        check_count = 1 if top_1_only else len(truth_set)
        truth_set_str = set(str(n) for n in truth_set)

        correct = 0.0
        for j in range(min(check_count, len(predictions))):
            if str(predictions[j]) in truth_set_str:
                correct += 1.0

        return correct / check_count

    def process_samples_batch(self, raw_samples):
        """
        Process a batch of samples and calculate aggregate metrics.

        This is the main entry point for batch processing after a task completes.

        Args:
            raw_samples: List of Sample objects from the sampler

        Returns:
            dict with aggregate metrics
        """
        if not raw_samples:
            return self._empty_metrics()

        # Extract timing data
        latencies = []
        service_times = []
        successful = 0
        failed = 0
        recall_values_k = []
        recall_values_1 = []

        for sample in raw_samples:
            if hasattr(sample, 'latency'):
                latencies.append(sample.latency * 1000)  # Convert to ms
            if hasattr(sample, 'service_time'):
                service_times.append(sample.service_time * 1000)  # Convert to ms

            # Check for success
            meta = getattr(sample, 'request_meta_data', {}) or {}
            if meta.get('success', True):
                successful += 1
            else:
                failed += 1

            # Extract recall if present
            if 'recall@k' in meta:
                recall_values_k.append(meta['recall@k'])
            if 'recall@1' in meta:
                recall_values_1.append(meta['recall@1'])

            # Calculate recall from candidate_ids if provided
            if 'candidate_ids' in meta and 'query_index' in meta:
                recall = self.calculate_recall(meta['candidate_ids'], meta['query_index'])
                if recall:
                    recall_values_k.append(recall['recall@k'])
                    recall_values_1.append(recall['recall@1'])

        # Calculate percentiles
        latencies.sort()
        service_times.sort()

        return {
            'total_ops': len(raw_samples),
            'successful_ops': successful,
            'failed_ops': failed,
            'latency_p50_ms': self._percentile(latencies, 50),
            'latency_p90_ms': self._percentile(latencies, 90),
            'latency_p99_ms': self._percentile(latencies, 99),
            'latency_avg_ms': sum(latencies) / len(latencies) if latencies else 0,
            'service_time_p50_ms': self._percentile(service_times, 50),
            'service_time_p90_ms': self._percentile(service_times, 90),
            'service_time_p99_ms': self._percentile(service_times, 99),
            'recall@k': sum(recall_values_k) / len(recall_values_k) if recall_values_k else None,
            'recall@1': sum(recall_values_1) / len(recall_values_1) if recall_values_1 else None,
            'error_rate': failed / len(raw_samples) if raw_samples else 0,
        }

    def _percentile(self, sorted_list, p):
        """Calculate percentile from a sorted list."""
        if not sorted_list:
            return 0
        idx = int(len(sorted_list) * p / 100)
        idx = min(idx, len(sorted_list) - 1)
        return sorted_list[idx]

    def _empty_metrics(self):
        """Return empty metrics dict."""
        return {
            'total_ops': 0,
            'successful_ops': 0,
            'failed_ops': 0,
            'latency_p50_ms': 0,
            'latency_p90_ms': 0,
            'latency_p99_ms': 0,
            'latency_avg_ms': 0,
            'service_time_p50_ms': 0,
            'service_time_p90_ms': 0,
            'service_time_p99_ms': 0,
            'recall@k': None,
            'recall@1': None,
            'error_rate': 0,
        }

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
            client_processing_time, processing_time, throughput, ops, ops_unit, time_period, task_progress,
            dependent_timing=None):
        try:
            self.q.put_nowait(
                DefaultSample(client_id, absolute_time, request_start, self.start_timestamp, task, sample_type, meta_data,
                       latency, service_time, client_processing_time, processing_time, throughput, ops, ops_unit, time_period,
                       task_progress, dependent_timing))
        except queue.Full:
            self.logger.warning("Dropping sample for [%s] due to a full sampling queue.", task.operation.name)

class ProfileMetricsSampler(Sampler):
    """
    Encapsulates management of gathered profile metrics samples.
    """

    def add(self, task, client_id, sample_type, meta_data, absolute_time, request_start, time_period, task_progress,
            dependent_timing=None):
        try:
            self.q.put_nowait(
                ProfileMetricsSample(client_id, absolute_time, request_start, self.start_timestamp, task, sample_type, meta_data,
                       time_period, task_progress, dependent_timing))
        except queue.Full:
            self.logger.warning("Dropping sample for [%s] due to a full sampling queue.", task.operation.name)


class Sample:
    """
    Basic information used by metrics store to keep track of samples
    """
    def __init__(self, client_id, absolute_time, request_start, task_start, task, sample_type, request_meta_data,
                time_period, task_progress, dependent_timing=None):
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
        self.task_progress = task_progress

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
    def __init__(self, client_id, absolute_time, request_start, task_start, task, sample_type, request_meta_data, latency,
                 service_time, client_processing_time, processing_time, throughput, total_ops, total_ops_unit, time_period,
                 task_progress, dependent_timing=None):
        super().__init__(client_id, absolute_time, request_start, task_start, task, sample_type, request_meta_data, time_period, task_progress, dependent_timing)
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
                             self.total_ops_unit, self.time_period, self.task_progress, None)

    def __repr__(self, *args, **kwargs):
        return f"[{self.absolute_time}; {self.relative_time}] [client [{self.client_id}]] [{self.task}] " \
               f"[{self.sample_type}]: [{self.latency}s] request latency, [{self.service_time}s] service time, " \
               f"[{self.total_ops} {self.total_ops_unit}]"

class ProfileMetricsSample(Sample):
    """
    Stores the profile metrics to later put into the metrics store
    """

    @property
    def dependent_timings(self):
        if self._dependent_timing:
            for t in self._dependent_timing:
                yield ProfileMetricsSample(self.client_id, t["absolute_time"], t["request_start"], self.task_start, self.task,
                             self.sample_type, self.request_meta_data, self.time_period, self.task_progress, None)


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
