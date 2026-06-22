# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Milvus client implementation."""

from osbenchmark.database.clients.milvus.client import MilvusClientFactory
from osbenchmark.database.clients.milvus.runners import register_milvus_runners

__all__ = ["MilvusClientFactory", "register_milvus_runners"]
