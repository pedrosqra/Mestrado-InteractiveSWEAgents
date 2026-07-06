#!/bin/bash
export PATH="/root/.local/bin:$PATH"
cd /root/InteractiveSWEAgents

MODELS=("qwen_32b")
SIMULATORS=("gpt-4o-mini")

(
  while true; do
    sleep 600
    DISK_USAGE=$(df / | tail -1 | awk "{print \$5}" | sed "s/%//")
    if [ "$DISK_USAGE" -gt 80 ]; then
        docker container prune -f > /dev/null 2>&1
        docker image prune -a -f --filter "until=24h" > /dev/null 2>&1
    fi
  done
) &
CLEANER_PID=$!
trap "kill $CLEANER_PID" EXIT

for MODEL in "${MODELS[@]}"; do
    for SIMULATOR in "${SIMULATORS[@]}"; do
        EXP_NOTE="v0.20.0-no-hint-${SIMULATOR}-32B-run_1"
        
        poetry run python evaluation/benchmarks/swe_bench/interact_run_infer.py \
          --agent-cls CodeActAgent \
          --llm-config $MODEL \
          --max-iterations 5 \
          --eval-num-workers 1 \
          --eval-note $EXP_NOTE \
          --dataset princeton-nlp/SWE-bench_Lite \
          --split test \
          --simulator_model $SIMULATOR
          
    done
done
