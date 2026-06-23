#!/bin/bash
cd /root/InteractiveSWEAgents
export PYTHONPATH="/root/InteractiveSWEAgents:$PYTHONPATH"

# Ativando poetry
source .venv/bin/activate

LLM_CONFIG="qwen_32b"
SIMULATOR="gemini-flash-latest"
EXP_NAME="v0.20.0-no-hint-${SIMULATOR}-${LLM_CONFIG}-run_1"

# Forçar salvamento na pasta organizada RQ3_run
export EVAL_OUTPUT_DIR="evaluation/evaluation_outputs/RQ3_run/${SIMULATOR}/${LLM_CONFIG}"
mkdir -p "$EVAL_OUTPUT_DIR"

echo "=========================================================================="
echo "🚀 INICIANDO BATERIA COM SIMULADOR: ${SIMULATOR} E MODELO: ${LLM_CONFIG}"
echo "=========================================================================="

nohup python evaluation/benchmarks/swe_bench/interact_run_infer.py \
  --agent-cls CodeActAgent \
  --llm-config "$LLM_CONFIG" \
  --max-iterations 5 \
  --eval-num-workers 1 \
  --eval-note "$EXP_NAME" \
  --dataset princeton-nlp/SWE-bench_Lite \
  --split test \
  --eval-n-limit 30 \
  --simulator_model "$SIMULATOR" > "rq3_${LLM_CONFIG}_${SIMULATOR}_evaluation.log" 2>&1 &

echo "Experimento iniciado em background! Logs em rq3_${LLM_CONFIG}_${SIMULATOR}_evaluation.log"
