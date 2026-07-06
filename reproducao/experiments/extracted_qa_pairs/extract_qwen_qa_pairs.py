"""
extract_qwen_qa_pairs.py
------------------------
Extrai pares pergunta-resposta (QA pairs) das 8 baterias da família Qwen2.5 Coder
e salva no formato padrão do projeto (mesmo formato dos arquivos claude_sonnet_qa_pairs.json,
haiku_qa_pairs.json, etc.) para uso nas análises de Distância do Cosseno e LLM-as-a-Judge.

Formato de saída (por arquivo):
[
  {
    "instance_id": "astropy__astropy-14309",
    "qa_pairs": [
      ["pergunta do agente", "resposta do simulador"],
      ...
    ]
  },
  ...
]

Uso:
    python extract_qwen_qa_pairs.py

Os arquivos serão salvos nesta mesma pasta (extracted_qa_pairs/).
"""

import json
from pathlib import Path

# ─── Configuração das baterias ────────────────────────────────────────────────
FINAIS = Path(__file__).resolve().parents[2] / "FINAIS"  # reproducao/FINAIS

BATERIAS = {
    "qwen_1_5b_gpt":    "Qwen2.5-Coder-1.5B-Instruct_maxiter_5_N_v0.20.0-no-hint-v0.20.0-no-hint-gpt-run_1-run_1",
    "qwen_1_5b_gemini": "Qwen2.5-Coder-1.5B-Instruct_maxiter_5_N_v0.20.0-no-hint-gemini-3.5-flash-1.5B-run_1",
    "qwen_7b_gpt":      "Qwen2.5-Coder-7B-Instruct-4bit_maxiter_5_N_v0.20.0-no-hint-gpt-4o-mini-7B-run_1",
    "qwen_7b_gemini":   "Qwen2.5-Coder-7B-Instruct-4bit_maxiter_5_N_v0.20.0-no-hint-gemini-3.5-flash-7B-run_1",
    "qwen_14b_gpt":     "Qwen2.5-Coder-14B-Instruct-4bit_maxiter_5_N_v0.20.0-no-hint-gpt-4o-mini-run_1",
    "qwen_14b_gemini":  "Qwen2.5-Coder-14B-Instruct-4bit_maxiter_5_N_v0.20.0-no-hint-gemini-3.5-flash-14B-run_1",
    "qwen_32b_gpt":     "qwen-2.5-coder-32b-instruct_maxiter_5_N_v0.20.0-no-hint-gpt-4o-mini-32B-run_1",
    "qwen_32b_gemini":  "qwen-2.5-coder-32b-instruct_maxiter_5_N_v0.20.0-no-hint-gemini-3.5-flash-32B-run_1",
}

# APENAS mensagens de sistema do framework — NÃO são respostas do simulador
# "I don't have that information" É uma resposta real do simulador (dado válido!)
SYSTEM_NOISE = [
    "<uploaded_files>",
    "Please continue working on the task",
    "Do NOT ask for more help",
    "ENVIRONMENT REMINDER",
]

def is_system_noise(text: str) -> bool:
    return any(noise in text for noise in SYSTEM_NOISE)


def extract_qa_pairs(history: list) -> list:
    """
    Extrai pares (pergunta_do_agente, resposta_do_simulador) do histórico de uma issue.

    O framework OpenHands salva as mensagens com os campos:
      - source: "agent" | "user"
      - action: "message" | "read" | "run" | "finish" | ...
      - message: texto da mensagem

    Uma pergunta válida é uma MessageAction do agente com texto substancial.
    Uma resposta válida é qualquer mensagem do simulador que não seja mensagem
    de sistema do framework (incluindo "I don't have that information", que é
    uma resposta legítima do simulador indicando falta de contexto).
    """
    qa_pairs = []
    pending_question = None

    for msg in history:
        source = msg.get("source", "")
        action = msg.get("action", "")
        text = msg.get("message", "").strip()

        # Pergunta: agente envia uma MessageAction com texto real
        if source == "agent" and action == "message" and len(text) > 30:
            pending_question = text

        # Resposta: simulador responde com qualquer coisa que não seja ruído do sistema
        elif source == "user" and pending_question and not is_system_noise(text) and len(text) > 5:
            qa_pairs.append([pending_question, text])
            pending_question = None

    return qa_pairs


def process_battery(name: str, folder_name: str) -> tuple:
    path = FINAIS / folder_name / "output.jsonl"

    if not path.exists():
        print(f"   Arquivo não encontrado: {path}")
        return name, [], {}

    results = []
    stats = {"total_issues": 0, "issues_com_qa": 0, "total_pares": 0}

    with open(path) as f:
        for line in f:
            issue = json.loads(line)
            instance_id = issue.get("instance_id", "")
            history = issue.get("history", [])

            qa_pairs = extract_qa_pairs(history)

            results.append({
                "instance_id": instance_id,
                "qa_pairs": qa_pairs
            })

            stats["total_issues"] += 1
            if qa_pairs:
                stats["issues_com_qa"] += 1
                stats["total_pares"] += len(qa_pairs)

    return name, results, stats


def main():
    output_dir = Path(__file__).parent
    print("=" * 60)
    print("EXTRAÇÃO DE QA PAIRS — Família Qwen2.5 Coder")
    print("=" * 60)

    for name, folder_name in BATERIAS.items():
        print(f"\nProcessando: {name}")
        battery_name, results, stats = process_battery(name, folder_name)

        if not results:
            continue

        output_file = output_dir / f"{battery_name}_qa_pairs.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        pct = 100 * stats["issues_com_qa"] / stats["total_issues"] if stats["total_issues"] else 0
        print(f"   Issues processadas: {stats['total_issues']}/30")
        print(f"   Issues com QA:      {stats['issues_com_qa']} ({pct:.0f}%)")
        print(f"   Total de pares:     {stats['total_pares']}")
        print(f"   Salvo em:           {output_file.name}")

    print("\n" + "=" * 60)
    print("Extração concluída!")
    print("Próximos passos:")
    print("  1. cosine_distance/  calcular ganho de informação")
    print("  2. llm_as_judge/     avaliar qualidade das perguntas")
    print("=" * 60)


if __name__ == "__main__":
    main()
