#!/bin/bash
curl -X 'POST' -H 'Content-Type:application/json' -d @sptrans_0_spec.json localhost:8090/druid/indexer/v1/task
curl -X 'POST' -H 'Content-Type:application/json' -d @sptrans_1_spec.json localhost:8090/druid/indexer/v1/task
curl -X 'POST' -H 'Content-Type:application/json' -d @sptrans_2_spec.json localhost:8090/druid/indexer/v1/task
curl -X 'POST' -H 'Content-Type:application/json' -d @sptrans_3_spec.json localhost:8090/druid/indexer/v1/task
curl -X 'POST' -H 'Content-Type:application/json' -d @sptrans_4_spec.json localhost:8090/druid/indexer/v1/task
curl -X 'POST' -H 'Content-Type:application/json' -d @sptrans_5_spec.json localhost:8090/druid/indexer/v1/task
curl -X 'POST' -H 'Content-Type:application/json' -d @sptrans_6_spec.json localhost:8090/druid/indexer/v1/task
curl -X 'POST' -H 'Content-Type:application/json' -d @sptrans_7_spec.json localhost:8090/druid/indexer/v1/task
curl -X 'POST' -H 'Content-Type:application/json' -d @sptrans_8_spec.json localhost:8090/druid/indexer/v1/task
curl -X 'POST' -H 'Content-Type:application/json' -d @sptrans_9_spec.json localhost:8090/druid/indexer/v1/task
curl -X 'POST' -H 'Content-Type:application/json' -d @sptrans_10_spec.json localhost:8090/druid/indexer/v1/task
curl -X 'POST' -H 'Content-Type:application/json' -d @sptrans_11_spec.json localhost:8090/druid/indexer/v1/task