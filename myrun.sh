#!/bin/bash

declare -a gpus=(6 7)

num_gpus=${#gpus[@]}

declare -a parts=()

for ((i=0; i<num_gpus; i++)); do
    parts+=($i)
done

for i in "${!gpus[@]}"; do
  command="CUDA_VISIBLE_DEVICES=${gpus[$i]} python running_sh.py --chunk_id ${parts[$i]} --total_chunks $num_gpus"
  echo "Executing: $command"
  eval $command &
  sleep 10
done

wait