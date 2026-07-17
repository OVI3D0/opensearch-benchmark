---
layout: default
title: ClickHouse support
parent: OSB Use Cases
nav_order: 40
---

# ClickHouse support

OpenSearch Benchmark (OSB) can benchmark a [ClickHouse](https://clickhouse.com/)
server in addition to OpenSearch. ClickHouse support is **experimental** in 3.0.

## What it does

When you select the ClickHouse engine, OSB connects to ClickHouse over its HTTP
interface (via the `clickhouse-connect` client) and runs the workload's
operations as ClickHouse-native SQL. Bulk operations map to `INSERT`, and search
operations run the SQL `SELECT` supplied by the workload's test procedure.

## Install

ClickHouse support ships as an optional extra:

```bash
pip install opensearch-benchmark[clickhouse]
```

This installs the `clickhouse-connect` dependency alongside OSB.

## Run

Select the engine with `--database-type=clickhouse` and point OSB at the
ClickHouse HTTP endpoint using `--target-hosts`:

```bash
opensearch-benchmark execute-test \
  --database-type=clickhouse \
  --pipeline=benchmark-only \
  --target-hosts=localhost:8123 \
  --workload=<clickhouse-native-workload>
```

- Use the **HTTP port `8123`** (or `8443` for HTTPS). The native binary
  protocol port `9000` is not supported by the HTTP client and is rejected
  with a clear error.
- `--pipeline=benchmark-only` is required, since OSB does not provision
  ClickHouse.

## Workloads must be ClickHouse-native

ClickHouse does not understand OpenSearch query DSL. To benchmark ClickHouse you
must use a workload whose **test procedures ship ClickHouse-native SQL** — each
search operation body must provide a `sql` key (with optional `parameters` and
`settings`), for example:

```json
{
  "sql": "SELECT count(*) FROM {{index}} WHERE status = {status:UInt16}",
  "parameters": {"status": 200}
}
```

If a search body is missing the `sql` key, OSB raises an error pointing back to
this guide.

These ClickHouse-native workload assets (param sources and SQL test procedures)
are **not shipped in the core OSB repository** — they are provided separately
(for example in the workloads repository). A stock OpenSearch workload will not
run against ClickHouse unchanged.

## Related

- [Migrating to 3.0](../migrating-to-3.0.md)
- [Vespa support](vespa-support.md)
- [Milvus support](milvus-support.md)
