---
layout: default
title: Migrating to 3.0
parent: OSB
nav_order: 25
---

# Migrating to OpenSearch Benchmark 3.0

OpenSearch Benchmark (OSB) 3.0 introduces two user-visible changes:

1. A **terminology revert** back to the `execute-test` / `test-execution`
   vocabulary used before 2.x. The old `run` / `test-run` terms remain
   accepted as **deprecated aliases**, and existing data stays readable.
2. A new **`--database-type`** flag that lets OSB benchmark non-OpenSearch
   engines (Vespa, Milvus, ClickHouse) in addition to OpenSearch.

There is also a new **CloudWatch reporting datastore**. See the
[CloudWatch reporting datastore guide](user-guides/cloudwatch-datastore.md)
for details.

This guide summarizes what changed for existing 2.x users and how to migrate.

## Terminology revert (2.x → 3.0)

OSB 3.0 restores the `execute-test` / `test-execution` terminology. To keep
2.x automation working, the previous `run` / `test-run` terms are preserved as
deprecated aliases wherever practical, and both the on-disk and stored-datastore
formats remain **readable** so existing benchmark history is not lost.

### Subcommand

| 2.x | 3.0 | Status |
| :-- | :-- | :-- |
| `opensearch-benchmark run ...` | `opensearch-benchmark execute-test ...` | `run` and `execute` still work as **deprecated aliases** and print a deprecation warning; update scripts to `execute-test`. |

Running `opensearch-benchmark run ...` (or `execute`) emits:

```
The 'run' subcommand is deprecated; use 'execute-test' instead.
```

The command still executes normally after the warning.

### CLI flags

| 2.x flag | 3.0 flag | Status |
| :-- | :-- | :-- |
| `--test-run-id` | `--test-execution-id` | Old spelling kept as an alias. |
| `list test-runs` | `list test-executions` | Old `test-runs` positional still accepted. |
| `--test-runs` (aggregate) | `--test-executions` (aggregate) | Old spelling kept as an alias. |

### On-disk layout

| 2.x | 3.0 |
| :-- | :-- |
| `~/.benchmark/benchmarks/test-runs/<id>/test_run.json` | `~/.benchmark/benchmarks/test-executions/<id>/test_execution.json` |

New test executions are written under `test-executions/`. OSB still **reads**
the pre-3.0 `test-runs/` directories (and `test_run.json` /
`aggregated_test_run.json` files), so historical local runs remain visible to
`list test-executions`, `compare`, and `generate` without any manual migration.

### Stored fields (OpenSearch / CloudWatch / on-disk documents)

| 2.x field | 3.0 field |
| :-- | :-- |
| `test-run-id` | `test-execution-id` |
| `test-run-timestamp` | `test-execution-timestamp` |

New documents are written with the `test-execution-*` field names, and the
metrics store index prefix moves from `benchmark-test-runs-*` to
`benchmark-test-executions-*`. Reads accept **both** the old and new field
names and **both** index prefixes, so metrics/results already stored in
OpenSearch, CloudWatch, or on disk stay readable. No re-indexing or data
migration is required.

### What you should do

- Update scripts and CI to use `execute-test` instead of `run` / `execute`.
- Update any flags from `--test-run-id` to `--test-execution-id` and
  `list test-runs` to `list test-executions` (the old forms keep working for now).
- No action is required for existing data — old directories, indices, and
  document fields remain readable.

## New: `--database-type`

OSB 3.0 adds a `--database-type` flag to `execute-test` that selects the target
database engine:

```
opensearch-benchmark execute-test --database-type=clickhouse --pipeline=benchmark-only \
  --target-hosts=localhost:8123 --workload=<engine-native-workload>
```

| Value | Engine | Extra to install |
| :-- | :-- | :-- |
| `opensearch` (default) | OpenSearch | (built in) |
| `vespa` | Vespa | `pip install opensearch-benchmark[vespa]` |
| `milvus` | Milvus | `pip install opensearch-benchmark[milvus]` |
| `clickhouse` | ClickHouse | `pip install opensearch-benchmark[clickhouse]` |

The default is `opensearch`, so existing OpenSearch benchmarks are unaffected.
See the per-engine user guides:

- [ClickHouse support](user-guides/clickhouse-support.md)
- [Vespa support](user-guides/vespa-support.md)
- [Milvus support](user-guides/milvus-support.md)

## Engine maturity and workload requirements (important)

The non-OpenSearch engines (Vespa, Milvus, ClickHouse) are **experimental** in
3.0. They cannot run end-to-end against a stock OpenSearch workload:

- Each engine requires **engine-native workload assets** — param sources and
  test procedures whose bodies target that engine (for example,
  ClickHouse-native SQL test procedures rather than OpenSearch query DSL). These
  assets are **not shipped in the core OSB repository**; they are provided
  separately (for example, in the workloads repository or alongside the engine
  integration).
- **Vespa vectorsearch is currently search-only** — the shipped Vespa
  vectorsearch integration covers the query path, not corpus ingestion via the
  stock workload.

Treat these engines as experimental and expect to supply engine-specific
workload assets when benchmarking them.
