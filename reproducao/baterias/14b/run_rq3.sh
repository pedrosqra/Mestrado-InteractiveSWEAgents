#!/bin/bash
# run_rq3.sh

# Modelos e simuladores
MODELS=("qwen_14b")
SIMULATORS=("gpt-4o-mini")

# NOTA (pós-experimento): este valor é repassado ao wrapper interact_run_infer.sh como
# --dataset (só um rótulo pra nomear a pasta de saída) E como --csv_file (que é quem de fato
# carrega os dados). Sem --csv_file, o interact_run_infer.py cairia no default
# (evaluation/benchmarks/swe_bench/data/full_summaries_verified.xlsx, um dataset bem maior) e só
# ficaria restrito às 30 instâncias da amostra do Passo 4 se o filtro "selected_ids" em
# evaluation/benchmarks/swe_bench/config.toml (gerado pelo generate_sample.py, não versionado no
# git) existisse na máquina. Com --csv_file explícito no wrapper, a amostra certa é garantida
# mesmo sem esse arquivo lateral.
DATASET_PATH="data/sample_30_underspecified.csv"
MAX_ITER=5

# ==========================================
# DAEMON DE LIMPEZA DE DISCO AUTOMATICA
# Roda a cada 10 minutos invisivel em background
# ==========================================
(
  while true; do
    sleep 600
    DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$DISK_USAGE" -gt 80 ]; then
        docker container prune -f > /dev/null 2>&1
        docker image prune -a -f --filter "until=24h" > /dev/null 2>&1
    fi
  done
) &
CLEANER_PID=$!

# Garante que o limpador morre se o script for cancelado
trap "kill $CLEANER_PID" EXIT

for sim in "${SIMULATORS[@]}"; do
    # Limpa o nome do simulador caso tenha barras (ex: openai/gpt-4o-mini)
    SIM_NAME=$(echo "$sim" | cut -d'/' -f2)

    for model in "${MODELS[@]}"; do
        echo "================================================="
        echo "Executando: Agent=${model} | Sim=${sim}"
        echo "================================================="

        # Define a pasta de saída organizada
        export EVAL_OUTPUT_DIR="evaluation/evaluation_outputs/RQ3_run/${SIM_NAME}/${model}"
        export EXP_NAME="${SIM_NAME}"
        mkdir -p "$EVAL_OUTPUT_DIR"

        # Chama o script do SWE-Bench
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
