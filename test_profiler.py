#!/usr/bin/env python3
"""
Quick test to verify the profiling infrastructure works correctly.
Run this before using profiling in a full benchmark.
"""

import time
import json
from osbenchmark import profiler


def test_profiler():
    """Test basic profiling functionality."""
    print("Testing OSB Profiler...")
    print("=" * 60)

    # Enable profiler
    profiler.enable()
    print("✓ Profiler enabled")

    # Test 1: Context manager
    print("\nTest 1: Context manager profiling")
    for i in range(100):
        with profiler.ProfileContext("test_context_manager"):
            time.sleep(0.001)  # 1ms sleep
    print("  ✓ Profiled 100 iterations with context manager")

    # Test 2: Manual start/stop
    print("\nTest 2: Manual start/stop profiling")
    for i in range(50):
        profiler.start("test_manual")
        time.sleep(0.002)  # 2ms sleep
        profiler.stop("test_manual")
    print("  ✓ Profiled 50 iterations manually")

    # Test 3: Decorator
    print("\nTest 3: Decorator profiling")
    @profiler.profile("test_decorator")
    def decorated_function():
        time.sleep(0.0015)  # 1.5ms sleep

    for i in range(75):
        decorated_function()
    print("  ✓ Profiled 75 iterations with decorator")

    # Test 4: Nested profiling
    print("\nTest 4: Nested profiling")
    for i in range(25):
        with profiler.ProfileContext("outer_operation"):
            time.sleep(0.001)
            with profiler.ProfileContext("inner_operation"):
                time.sleep(0.001)
            time.sleep(0.001)
    print("  ✓ Profiled 25 nested operations")

    # Get statistics
    print("\n" + "=" * 60)
    print("Profiling Results:")
    print("=" * 60)

    stats = profiler.get_stats()

    # Display results
    print(f"\n{'Operation':<30} {'Count':>8} {'Total(ms)':>12} {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}")
    print("-" * 90)

    for name, stat in sorted(stats.items()):
        print(f"{name:<30} {stat.count:>8} {stat.total_time*1000:>12.2f} "
              f"{stat.avg_time*1000:>10.3f} {stat.min_time*1000:>10.3f} {stat.max_time*1000:>10.3f}")

    # Verify expectations
    print("\n" + "=" * 60)
    print("Verification:")
    print("=" * 60)

    errors = []

    # Check test_context_manager
    if "test_context_manager" in stats:
        cm_stats = stats["test_context_manager"]
        if cm_stats.count != 100:
            errors.append(f"test_context_manager: expected 100 calls, got {cm_stats.count}")
        if cm_stats.avg_time < 0.0005 or cm_stats.avg_time > 0.005:
            errors.append(f"test_context_manager: avg time {cm_stats.avg_time*1000:.2f}ms seems wrong (expected ~1ms)")
        print(f"✓ test_context_manager: {cm_stats.count} calls, avg {cm_stats.avg_time*1000:.2f}ms")
    else:
        errors.append("test_context_manager not found in results")

    # Check test_manual
    if "test_manual" in stats:
        manual_stats = stats["test_manual"]
        if manual_stats.count != 50:
            errors.append(f"test_manual: expected 50 calls, got {manual_stats.count}")
        if manual_stats.avg_time < 0.001 or manual_stats.avg_time > 0.01:
            errors.append(f"test_manual: avg time {manual_stats.avg_time*1000:.2f}ms seems wrong (expected ~2ms)")
        print(f"✓ test_manual: {manual_stats.count} calls, avg {manual_stats.avg_time*1000:.2f}ms")
    else:
        errors.append("test_manual not found in results")

    # Check test_decorator
    if "test_decorator" in stats:
        dec_stats = stats["test_decorator"]
        if dec_stats.count != 75:
            errors.append(f"test_decorator: expected 75 calls, got {dec_stats.count}")
        print(f"✓ test_decorator: {dec_stats.count} calls, avg {dec_stats.avg_time*1000:.2f}ms")
    else:
        errors.append("test_decorator not found in results")

    # Check nested operations
    if "outer_operation" in stats and "inner_operation" in stats:
        outer = stats["outer_operation"]
        inner = stats["inner_operation"]
        if outer.count != 25:
            errors.append(f"outer_operation: expected 25 calls, got {outer.count}")
        if inner.count != 25:
            errors.append(f"inner_operation: expected 25 calls, got {inner.count}")
        print(f"✓ Nested profiling: outer={outer.count}, inner={inner.count}")
    else:
        errors.append("Nested operations not found in results")

    # Write results to file
    print("\n" + "=" * 60)
    print("Writing results...")
    profiler.write_results("test_profiling_results.json")
    print("✓ Results written to test_profiling_results.json")

    # Verify JSON file
    try:
        with open("test_profiling_results.json") as f:
            results = json.load(f)

        print("\nJSON file contents:")
        print(f"  Total operations: {results['summary']['total_operations']}")
        print(f"  Profiled operations: {results['summary']['num_profiled_operations']}")
        print(f"  Total time: {results['summary']['total_time_s']:.3f}s")
        print("✓ JSON file valid and readable")
    except Exception as e:
        errors.append(f"Failed to read JSON file: {e}")

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print("❌ ERRORS FOUND:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ ALL TESTS PASSED!")
        print("\nThe profiling system is working correctly.")
        print("You can now use it in full OSB benchmarks.")
        return True


if __name__ == "__main__":
    success = test_profiler()
    exit(0 if success else 1)
