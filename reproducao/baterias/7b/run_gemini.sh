#!/bin/bash
export PATH="/root/.local/bin:$PATH"
cd /root/Mestrado-InteractiveSWEAgents

MODELS=("qwen_7b")
SIMULATORS=("gemini-flash-latest")

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
        EXP_NOTE="v0.20.0-no-hint-gemini-flash-latest-7B-run_1"

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
