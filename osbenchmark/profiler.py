# SPDX-License-Identifier: Apache-2.0
"""
Lightweight profiling infrastructure for OpenSearch Benchmark.

Tracks timing data across all components with min/max/avg for repeated operations.
Outputs results to a JSON file for analysis.

Usage:
    from osbenchmark import profiler

    # Method 1: Decorator for functions
    @profiler.profile("my_function")
    def my_function():
        ...

    # Method 2: Context manager for code blocks
    with profiler.ProfileContext("operation_name"):
        ...

    # Method 3: Manual start/stop
    profiler.start("operation_name")
    ...
    profiler.stop("operation_name")

    # At end of benchmark, write results
    profiler.write_results()
"""

import functools
import json
import logging
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class TimingStat:
    """Statistics for a profiled operation."""
    name: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.count if self.count > 0 else 0.0

    def add_timing(self, duration: float):
        """Add a timing measurement."""
        self.count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'count': self.count,
            'total_time_s': round(self.total_time, 6),
            'avg_time_ms': round(self.avg_time * 1000, 3),
            'min_time_ms': round(self.min_time * 1000, 3) if self.min_time != float('inf') else 0.0,
            'max_time_ms': round(self.max_time * 1000, 3),
        }


class Profiler:
    """
    Thread-safe profiler for OSB components.

    Tracks timing data for operations across the entire benchmark lifecycle.
    """

    def __init__(self):
        self.enabled = False
        self.stats: Dict[str, TimingStat] = {}
        self.lock = threading.RLock()  # Reentrant lock for nested profiling
        self.active_timers: Dict[int, Dict[str, float]] = defaultdict(dict)  # Per-thread active timers
        self.logger = logging.getLogger(__name__)

    def enable(self):
        """Enable profiling."""
        self.enabled = True
        self.logger.info("Profiler enabled")

    def disable(self):
        """Disable profiling."""
        self.enabled = False
        self.logger.info("Profiler disabled")

    def start(self, operation_name: str):
        """Start timing an operation."""
        if not self.enabled:
            return

        thread_id = threading.get_ident()
        self.active_timers[thread_id][operation_name] = time.perf_counter()

    def stop(self, operation_name: str):
        """Stop timing an operation and record the duration."""
        if not self.enabled:
            return

        end_time = time.perf_counter()
        thread_id = threading.get_ident()

        start_time = self.active_timers[thread_id].get(operation_name)
        if start_time is None:
            self.logger.warning(f"Profiler.stop() called for '{operation_name}' without matching start()")
            return

        duration = end_time - start_time
        del self.active_timers[thread_id][operation_name]

        with self.lock:
            if operation_name not in self.stats:
                self.stats[operation_name] = TimingStat(name=operation_name)
            self.stats[operation_name].add_timing(duration)

    def get_stats(self) -> Dict[str, TimingStat]:
        """Get all timing statistics."""
        with self.lock:
            return dict(self.stats)

    def reset(self):
        """Clear all collected stats."""
        with self.lock:
            self.stats.clear()
            self.active_timers.clear()
            self.logger.info("Profiler stats reset")

    def write_results(self, output_path: Optional[str] = None):
        """
        Write profiling results to a JSON file.

        :param output_path: Path to output file. If None, uses default location.
        """
        if not self.enabled:
            self.logger.info("Profiler disabled, skipping results write")
            return

        if output_path is None:
            # Default to OSB root directory
            output_path = os.path.join(os.getcwd(), "profiling_results.json")

        results = {
            'summary': {
                'total_operations': sum(s.count for s in self.stats.values()),
                'total_time_s': sum(s.total_time for s in self.stats.values()),
                'num_profiled_operations': len(self.stats),
            },
            'operations': {}
        }

        # Sort by total time descending
        sorted_stats = sorted(self.stats.values(), key=lambda s: s.total_time, reverse=True)

        for stat in sorted_stats:
            results['operations'][stat.name] = stat.to_dict()

        # Write to file
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Profiling results written to {output_path}")
            print(f"\n=== Profiling Results ===")
            print(f"Output: {output_path}")
            print(f"Total operations profiled: {results['summary']['total_operations']:,}")
            print(f"Total time tracked: {results['summary']['total_time_s']:.2f}s")
            print(f"\nTop 10 operations by total time:")
            for i, stat in enumerate(sorted_stats[:10], 1):
                print(f"  {i:2d}. {stat.name:50s} {stat.total_time:8.3f}s ({stat.count:,} calls)")
        except Exception as e:
            self.logger.error(f"Failed to write profiling results: {e}")


# Global profiler instance
_profiler = Profiler()


def enable():
    """Enable global profiler."""
    _profiler.enable()


def disable():
    """Disable global profiler."""
    _profiler.disable()


def start(operation_name: str):
    """Start timing an operation."""
    _profiler.start(operation_name)


def stop(operation_name: str):
    """Stop timing an operation."""
    _profiler.stop(operation_name)


def reset():
    """Reset profiler stats."""
    _profiler.reset()


def write_results(output_path: Optional[str] = None):
    """Write profiling results to file."""
    _profiler.write_results(output_path)


def get_stats() -> Dict[str, TimingStat]:
    """Get all timing statistics."""
    return _profiler.get_stats()


@contextmanager
def ProfileContext(operation_name: str):
    """
    Context manager for profiling a code block.

    Usage:
        with ProfileContext("my_operation"):
            # code to profile
            ...
    """
    _profiler.start(operation_name)
    try:
        yield
    finally:
        _profiler.stop(operation_name)


def profile(operation_name: Optional[str] = None):
    """
    Decorator for profiling a function.

    Usage:
        @profile("function_name")
        def my_function():
            ...

        # Or use function name automatically:
        @profile()
        def my_function():
            ...
    """
    def decorator(func):
        name = operation_name or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            _profiler.start(name)
            try:
                return func(*args, **kwargs)
            finally:
                _profiler.stop(name)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            _profiler.start(name)
            try:
                return await func(*args, **kwargs)
            finally:
                _profiler.stop(name)

        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    # Handle both @profile and @profile()
    if callable(operation_name):
        func = operation_name
        operation_name = None
        return decorator(func)
    else:
        return decorator


# Import asyncio after function definitions to avoid circular import
import asyncio
