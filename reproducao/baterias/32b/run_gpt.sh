#!/bin/bash
export PATH="/root/.local/bin:$PATH"
cd /root/Mestrado-InteractiveSWEAgents

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

        # NOTA (pós-experimento): o --dataset abaixo antes apontava para "princeton-nlp/SWE-bench_Lite",
        # mas o resultado real já era restrito às mesmas 30 instâncias da amostra do Passo 4, porque
        # o interact_run_infer.py sempre filtra o dataset carregado pelos "selected_ids" de um
        # evaluation/benchmarks/swe_bench/config.toml local (gerado pelo generate_sample.py e não
        # versionado no git). Isso só torna essa restrição explícita no próprio script.
        poetry run python evaluation/benchmarks/swe_bench/interact_run_infer.py \
          --agent-cls CodeActAgent \
          --llm-config $MODEL \
          --max-iterations 5 \
          --eval-num-workers 1 \
          --eval-note $EXP_NOTE \
          --dataset data/sample_30_underspecified.csv \
          --split test \
          --simulator_model $SIMULATOR
          
    done
done
