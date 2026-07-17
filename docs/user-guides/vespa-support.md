---
layout: default
title: Vespa support
parent: OSB Use Cases
nav_order: 41
---

# Vespa support

OpenSearch Benchmark (OSB) can benchmark a [Vespa](https://vespa.ai/) cluster in
addition to OpenSearch. Vespa support is **experimental** in 3.0.

## What it does

When you select the Vespa engine, OSB connects to Vespa (via the `pyvespa`
client) and runs the workload's operations against Vespa's document and query
APIs. The current integration focuses on vector search: Vespa **vectorsearch is
search-only** in 3.0 (the query path is exercised; corpus ingestion via the
stock workload is not covered).

## Install

Vespa support ships as an optional extra:

```bash
pip install opensearch-benchmark[vespa]
```

This installs the `pyvespa` dependency alongside OSB.

## Run

Select the engine with `--database-type=vespa` and point OSB at your Vespa
endpoint with `--target-hosts`:

```bash
opensearch-benchmark execute-test \
  --database-type=vespa \
  --pipeline=benchmark-only \
  --target-hosts=<vespa-endpoint> \
  --workload=<vespa-native-workload>
```

`--pipeline=benchmark-only` is required, since OSB does not provision Vespa.

## Workloads must be Vespa-native

Vespa does not understand OpenSearch query DSL. Benchmarking Vespa requires
**Vespa-native workload assets** — vector-search param sources and test
procedures whose bodies target Vespa. These assets are **not shipped in the core
OSB repository**; they are provided separately (for example in the workloads
repository). A stock OpenSearch workload will not run against Vespa unchanged.

## Related

- [Migrating to 3.0](../migrating-to-3.0.md)
- [ClickHouse support](clickhouse-support.md)
- [Milvus support](milvus-support.md)
