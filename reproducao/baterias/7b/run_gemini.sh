#!/bin/bash
export PATH="/root/.local/bin:$PATH"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY}"
cd /root/InteractiveSWEAgents

MODELS=("qwen_7b")
SIMULATORS=("openrouter/google/gemini-3.5-flash")

(
  while true; do
    sleep 600
    DISK_USAGE=$(df / | tail -1 | awk "{print \$5}" | sed "s/%//")
    if [ "$DISK_USAGE" -gt 80 ]; then
        # Mantendo apenas container prune para não apagar as imagens base (que corrompe o SWE-Bench)
        docker container prune -f > /dev/null 2>&1
    fi
  done
) &
CLEANER_PID=$!
trap "kill $CLEANER_PID" EXIT

for MODEL in "${MODELS[@]}"; do
    for SIMULATOR in "${SIMULATORS[@]}"; do
        # Nome da rodada limpo (sem as barras do openrouter para não dar erro no Linux)
        EXP_NOTE="v0.20.0-no-hint-gemini-3.5-flash-7B-run_1"
        
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
