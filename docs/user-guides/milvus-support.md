---
layout: default
title: Milvus support
parent: OSB Use Cases
nav_order: 42
---

# Milvus support

OpenSearch Benchmark (OSB) can benchmark a [Milvus](https://milvus.io/) vector
database in addition to OpenSearch. Milvus support is **experimental** in 3.0.

## What it does

When you select the Milvus engine, OSB connects to Milvus (via the `pymilvus`
client) and runs the workload's operations against Milvus's collection and
vector-search APIs.

## Install

Milvus support ships as an optional extra:

```bash
pip install opensearch-benchmark[milvus]
```

This installs the `pymilvus` dependency alongside OSB.

## Run

Select the engine with `--database-type=milvus` and point OSB at your Milvus
endpoint with `--target-hosts`:

```bash
opensearch-benchmark execute-test \
  --database-type=milvus \
  --pipeline=benchmark-only \
  --target-hosts=<milvus-endpoint> \
  --workload=<milvus-native-workload>
```

`--pipeline=benchmark-only` is required, since OSB does not provision Milvus.

## Workloads must be Milvus-native

Milvus does not understand OpenSearch query DSL. Benchmarking Milvus requires
**Milvus-native workload assets** — vector-search param sources and test
procedures whose bodies target Milvus. These assets are **not shipped in the
core OSB repository**; they are provided separately (for example in the
workloads repository). A stock OpenSearch workload will not run against Milvus
unchanged.

## Related

- [Migrating to 3.0](../migrating-to-3.0.md)
- [ClickHouse support](clickhouse-support.md)
- [Vespa support](vespa-support.md)
