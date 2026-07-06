import os
"""
generate_qwen_cosine_distances.py
----------------------------------
Calcula a Distância do Cosseno (Ganho de Informação) para as 8 baterias
da família Qwen2.5 Coder, usando os qa_pairs gerados pelo extract_qwen_qa_pairs.py.

Diferenças em relação ao generate_cosine_distances.py original:
  - Itera sobre todas as 8 baterias automaticamente
  - Usa sample_30_underspecified.csv em vez do full_summaries_verified.xlsx
  - Issues sem qa_pairs (ex: 1.5B que não interagiu) recebem difference_score = 0.0

Uso:
    cd experiments/cosine_distance/
    python generate_qwen_cosine_distances.py
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from litellm import embedding

# ─── Configuração ─────────────────────────────────────────────────────────────
API_KEY = os.environ.get("OPENAI_API_KEY")  # Coloque sua chave OpenAI aqui

BASE_DIR   = Path(__file__).parent.parent   # experiments/
QA_DIR     = BASE_DIR / 'extracted_qa_pairs'
OUTPUT_DIR = Path(__file__).parent          # cosine_distance/

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


# ─── Funções ──────────────────────────────────────────────────────────────────

def get_embedding(text):
    response = embedding(
        model='openai/text-embedding-3-small',
        input=[text],
        api_key=API_KEY,
    )
    return response['data'][0]['embedding']


def clean_answer(answer):
    phrases_to_remove = [
        "I don't have that information",
        "I don't have information about that",
        "I don't have that specific information",
        "I don't have details on that",
    ]
    for phrase in phrases_to_remove:
        answer = answer.replace(phrase, '')
    return answer.strip()


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
        
        ps_embedding = get_embedding(problem_statement)

        for q, a in entry['qa_pairs']:
            an = clean_answer(a)
            try:
                ps_a_embedding = get_embedding(problem_statement + ' ' + an)
            except Exception as e:
                print(f'     Erro ao gerar embedding ({instance_id}): {e}')
                continue

            sim = cosine_similarity([ps_embedding], [ps_a_embedding])[0][0]
            results.append({
                'instance_id':      instance_id,
                'similarity':       float(sim),
                'difference_score': float(1 - sim),
                'question':         q,
                'answer':           a,
                'problem_statement': problem_statement,
            })

    return results


def main():
    print('Carregando problem statements...')
    df = pd.read_csv(CSV_PATH)
    problem_statements = dict(zip(df['instance_id'], df['problem_statement']))
    print(f'  {len(problem_statements)} instâncias carregadas.\n')

    print('=' * 60)
    print('DISTÂNCIA DO COSSENO — Família Qwen2.5 Coder')
    print('=' * 60)

    for name in BATERIAS:
        print(f'\nProcessando: {name}')
        results = process_battery(name, problem_statements)

        if not results:
            print('   Nenhum resultado.')
            continue

        results_df = pd.DataFrame(results)
        out_path = OUTPUT_DIR / f'{name}_embedding_results.csv'
        results_df.to_csv(out_path, index=False)

        pares_reais = results_df[results_df['question'] != '']
        print(f'   Issues processadas:     {results_df["instance_id"].nunique()}/30')
        print(f'   Pares com resposta:     {len(pares_reais)}')
        print(f'   difference_score médio: {results_df["difference_score"].mean():.4f}')
        print(f'   Salvo em:               {out_path.name}')

    print('\n' + '=' * 60)
    print('Concluído!')
    print('=' * 60)


if __name__ == '__main__':
    main()
