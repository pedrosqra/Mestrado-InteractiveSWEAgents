#!/bin/bash
MODELS=("qwen_1_5b")
SIMULATORS=("gemini-flash-latest")

# NOTA (pós-experimento): este valor é repassado ao wrapper interact_run_infer.sh como
# (evaluation/benchmarks/swe_bench/data/full_summaries_verified.xlsx, um dataset bem maior) e só
# ficaria restrito às 30 instâncias da amostra do Passo 4 se o filtro "selected_ids" em
# evaluation/benchmarks/swe_bench/config.toml (gerado pelo generate_sample.py, não versionado no
# mesmo sem esse arquivo lateral.
DATASET_PATH="data/sample_30_underspecified.csv"
MAX_ITER=5

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

for sim in "${SIMULATORS[@]}"; do
    SIM_NAME=$(echo "$sim" | cut -d/ -f2)

    for model in "${MODELS[@]}"; do
        echo "================================================="
        echo "Executando: Agent=${model} | Sim=${sim}"
        echo "================================================="

        export EVAL_OUTPUT_DIR="evaluation/evaluation_outputs/outputs/data__sample_30_underspecified.csv-test/CodeActAgent/Qwen2.5-Coder-1.5B-Instruct_maxiter_5_N_v0.20.0-no-hint-gemini-3.5-flash-1.5B-run_1"
        export EXP_NAME="v0.20.0-no-hint-run_1"
        mkdir -p "$EVAL_OUTPUT_DIR"

        bash evaluation/benchmarks/swe_bench/scripts/interact_run_infer.sh \
            "$model" \
            HEAD \
            CodeActAgent \
            "" \
            "$MAX_ITER" \
            1 \
            "$DATASET_PATH" \
            test \
            1 \
            "$sim"
    done
done
