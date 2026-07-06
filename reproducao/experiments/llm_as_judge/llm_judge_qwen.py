import os
"""
llm_judge_qwen.py
------------------
Adaptação do llm_judge.py para avaliar as respostas (Answers) obtidas 
pelas 8 baterias da família Qwen2.5 Coder, utilizando o modelo GPT-4o como juiz.

O script calcula notas de 1 a 5 com base no nível de detalhamento e novidade
da resposta em relação ao problema original.

Uso:
    cd experiments/llm_as_judge/
    python3 llm_judge_qwen.py
"""

import json
import re
from pathlib import Path
import pandas as pd
import litellm

# ─── Configuração ─────────────────────────────────────────────────────────────
API_KEY = os.environ.get("OPENAI_API_KEY")  # Chave OpenAI

BASE_DIR = Path(__file__).parent.parent      # experiments/
QA_DIR = BASE_DIR / 'extracted_qa_pairs'
OUTPUT_DIR = Path(__file__).parent            # llm_as_judge/

CSV_PATH = Path(__file__).resolve().parents[3] / 'data' / 'sample_30_underspecified.csv'

BATERIAS = [
    'qwen_1_5b_gpt',
    'qwen_1_5b_gemini',
    'qwen_7b_gpt',
    'qwen_7b_gemini',
    'qwen_14b_gpt',
    'qwen_14b_gemini',
    'qwen_32b_gpt',
    'qwen_32b_gemini',
]


# ─── Função de Avaliação do Juiz ──────────────────────────────────────────────

def evaluate_answer(problem_statement, question, answer):
    """Avalia a resposta em uma escala de 1 a 5 usando GPT-4o."""
    prompt = f"""Evaluate the following answer on a scale of 1–5 based on how much new and relevant details it adds to the given GitHub issue, which would help make solving the issue easier. 

    **Original Issue:**
    {problem_statement}

    **Questions:**
    {question}

    **Answer:**
    {answer}

    **Evaluation Criteria:**
    - 1: Adds no new or relevant information.
    - 2: Adds minor details, but largely redundant or irrelevant.
    - 3: Adds some new information, but lacks specificity or relevance.
    - 4: Adds significant new and relevant information.
    - 5: Adds highly detailed and critical new information that addresses key gaps.

    Please provide your score inside <score></score> tags.

    **Score:**
    """
    
    response = litellm.completion(
        api_key=API_KEY,
        model="openai/gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant adept at nuanced evaluations."},
            {"role": "user", "content": prompt}
        ]
    )
    
    score_text = response['choices'][0]['message']['content'].strip()
    score_match = re.search(r'<score>(\d+)</score>', score_text)
    if score_match:
        try:
            score = int(score_match.group(1))
            if 1 <= score <= 5:
                return score
            else:
                raise ValueError("Score fora do intervalo de 1 a 5")
        except ValueError:
            print(f"   Score inválido retornado: {score_match.group(1)}")
    else:
        print(f"   Não foi possível extrair o score da resposta: {score_text}")
    return None


def process_battery(name, problem_statements):
    qa_path = QA_DIR / f'{name}_qa_pairs.json'
    if not qa_path.exists():
        print(f'   Arquivo não encontrado: {qa_path}')
        return []

    with open(qa_path) as f:
        qa_data = json.load(f)

    results = []

    for entry in qa_data:
        instance_id = entry['instance_id']
        if instance_id not in problem_statements:
            continue
        
        problem_statement = problem_statements[instance_id]
        
        # Se a issue não tem qa_pairs (ex: 1.5B), não há respostas a avaliar
        for q, a in entry['qa_pairs']:
            try:
                new_info_score = evaluate_answer(problem_statement, q, a)
            except Exception as e:
                print(f'     Erro ao avaliar com GPT-4o ({instance_id}): {e}')
                continue

            if new_info_score is not None:
                results.append({
                    'instance_id': instance_id,
                    'question': q,
                    'answer': a,
                    'problem_statement': problem_statement,
                    'new_information_score': new_info_score
                })

    return results


def main():
    print('Carregando problem statements...')
    df = pd.read_csv(CSV_PATH)
    problem_statements = dict(zip(df['instance_id'], df['problem_statement']))
    print(f'  {len(problem_statements)} instâncias carregadas.\n')

    print('=' * 60)
    print('LLM-AS-JUDGE EVALUATION — Família Qwen2.5 Coder')
    print('=' * 60)

    for name in BATERIAS:
        print(f'\nProcessando: {name}')
        results = process_battery(name, problem_statements)

        results_df = pd.DataFrame(results)
        out_path = OUTPUT_DIR / f'{name}_gpt4o_evaluation_results.csv'
        results_df.to_csv(out_path, index=False)

        if len(results_df) > 0:
            avg_score = results_df['new_information_score'].mean()
            print(f'   Pares avaliados: {len(results_df)}')
            print(f'   Score médio:     {avg_score:.2f}/5.0')
        else:
            print('   Processado (zero pares QA no modelo).')
        print(f'   Salvo em:         {out_path.name}')

    print('\n' + '=' * 60)
    print('Concluído!')
    print('=' * 60)


if __name__ == '__main__':
    main()
