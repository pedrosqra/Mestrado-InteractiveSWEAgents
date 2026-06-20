#!/bin/bash
# run_240_rounds.sh
# Automatização do Experimento RQ3: Escala de Elicitação (Qwen2.5 Coder)
# Fatores: 4 Agentes x 2 Simuladores x 30 Instâncias = 240 rodadas

MODELS=("qwen_1_5b" "qwen_7b" "qwen_14b" "qwen_32b")
SIMULATORS=("openai/gpt-4o-mini" "gemini-flash-latest")

# Caminho para o dataset amostrado (Fase 1)
DATASET_PATH="data/sample_30_underspecified.csv"

for sim in "${SIMULATORS[@]}"; do
  # Nome amigável para a pasta de resultados (ex: gpt-4o-mini)
  SIM_NAME=$(echo $sim | cut -d'/' -f2)
  
  for model in "${MODELS[@]}"; do
    echo "=========================================================="
    echo "🚀 Iniciando Experimento: Agente=$model | Simulador=$sim"
    echo "=========================================================="
    
    # Define o diretório de saída estruturado (Boas Práticas de Organização)
    export EVAL_OUTPUT_DIR="evaluation/evaluation_outputs/outputs/interactivity/${SIM_NAME}/${model}"
    mkdir -p $EVAL_OUTPUT_DIR

    # Executa o benchmark
    # Parâmetros: MODEL_CONFIG, COMMIT_HASH, AGENT, EVAL_LIMIT, MAX_ITER, NUM_WORKERS, DATASET, SPLIT, N_RUNS, SIMULATOR_MODEL
    bash evaluation/benchmarks/swe_bench/scripts/interact_run_infer.sh \
      $model \
      HEAD \
      CodeActAgent \
      30 \
      15 \
      1 \
      $DATASET_PATH \
      test \
      1 \
      $sim

    echo "✅ Concluído: $model com $sim"
    echo "----------------------------------------------------------"
  done
done

echo "🎉 Experimento Completo! Resultados em evaluation/evaluation_outputs/outputs/interactivity/"
