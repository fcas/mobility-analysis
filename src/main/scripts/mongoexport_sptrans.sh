#!/bin/bash
loops=12
count=25284071
batch_size=$((count / loops))

for i in {0..11}
do
    mongoexport --db sptrans --collection logs_avl --skip $((batch_size * i)) --limit $((batch_size)) --out sptrans$((i)).json
done
