#!/bin/bash
export PYTHONPATH="/root/InteractiveSWEAgents:$PYTHONPATH"

# Ativando poetry
cd /root/InteractiveSWEAgents
source .venv/bin/activate

# Rodar a avaliação com o simulador GPT
nohup python3 evaluation/benchmarks/swe_bench/interact_run_infer.py \
  --agent-cls CodeActAgent \
  --llm-config qwen_14b \
  --max-iterations 5 \
  --eval-num-workers 1 \
  --eval-note v0.20.0-gpt_simulator \
  --dataset princeton-nlp/SWE-bench_Lite \
  --split test \
  --simulator_model gpt-4o-mini > rq3_gpt_evaluation.log 2>&1 &

echo "Experimento RQ3 com o Simulador GPT iniciado em background!"
echo "Pode rodar o 'bash painel.sh' para acompanhar."
