import os
"""
num_ques_qwen.py
-----------------
Conta o número de perguntas dentro de cada MessageAction para as 8 baterias
da família Qwen2.5 Coder, usando GPT-4o — mesma metodologia do paper original.

Isso produz o avg_q comparável ao 6.02 reportado para Qwen 3 Coder no paper.

Uso:
    cd experiments/extracted_qa_pairs/
    python3 num_ques_qwen.py
"""

import json
import re
from pathlib import Path

import litellm
import pandas as pd

# ─── Configuração ─────────────────────────────────────────────────────────────
API_KEY = os.environ.get("OPENAI_API_KEY")  # Coloque sua chave OpenAI aqui

BASE_DIR = Path(__file__).parent                # extracted_qa_pairs/
OUTPUT_DIR = Path(__file__).parent              # mesmo diretório
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


# ─── Função idêntica ao original ──────────────────────────────────────────────

def count_questions(question):
    """Usa GPT-4o para contar o número de perguntas dentro de um MessageAction."""
    prompt = f"""Count the number of questions in the list of questions. 

    **Questions:**
    {question}

    Please provide your answer as an integer inside <question></question> tags.

    **Number of Questions:**
    """

    response = litellm.completion(
        api_key=API_KEY,
        model='openai/gpt-4o',
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant adept at nuanced evaluations.'},
            {'role': 'user',   'content': prompt},
        ]
    )

    score_text = response['choices'][0]['message']['content'].strip()
    score_match = re.search(r'<question>(\d+)</question>', score_text)
    if score_match:
        try:
            return int(score_match.group(1))
        except ValueError:
            print(f'   Valor inválido: {score_match.group(1)}')
    else:
        print(f'   Não foi possível extrair contagem: {score_text}')
    return None


def process_battery(name, problem_statements):
    qa_path = BASE_DIR / f'{name}_qa_pairs.json'
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

        # Issues sem QA (1.5B) 0 perguntas
        if not entry['qa_pairs']:
            results.append({
                'instance_id':      instance_id,
                'question':         '',
                'answer':           '',
                'problem_statement': problem_statement,
                'num_questions':    0,
            })
            continue

        for q, a in entry['qa_pairs']:
            try:
                n = count_questions(q)
            except Exception as e:
                print(f'     Erro na chamada GPT-4o ({instance_id}): {e}')
                continue

            results.append({
                'instance_id':      instance_id,
                'question':         q,
                'answer':           a,
                'problem_statement': problem_statement,
                'num_questions':    n,
            })

    return results


def main():
    print('Carregando problem statements...')
    df_csv = pd.read_csv(CSV_PATH)
    problem_statements = dict(zip(df_csv['instance_id'], df_csv['problem_statement']))
    print(f'  {len(problem_statements)} instâncias carregadas.\n')

    print('=' * 60)
    print('CONTAGEM DE PERGUNTAS — Família Qwen2.5 Coder')
    print('=' * 60)

    for name in BATERIAS:
        print(f'\nProcessando: {name}')
        results = process_battery(name, problem_statements)

        if not results:
            print('   Nenhum resultado.')
            continue

        results_df = pd.DataFrame(results)
        out_path = OUTPUT_DIR / f'{name}_num_ques.csv'
        results_df.to_csv(out_path, index=False)

        total_q = results_df['num_questions'].sum()
        n_issues = results_df['instance_id'].nunique()
        avg_q = total_q / n_issues if n_issues else 0

        print(f'   Issues processadas: {n_issues}/30')
        print(f'   Total de perguntas: {total_q}')
        print(f'   avg_q por issue:    {avg_q:.2f}')
        print(f'   Salvo em:           {out_path.name}')

    print('\n' + '=' * 60)
    print('Concluído!')
    print('=' * 60)

    # Resumo comparativo final
    print('\n=== RESUMO (comparável ao paper) ===')
    print(f'  {"Bateria":<22} {"avg_q":>8}')
    print('  ' + '-' * 32)
    for name in BATERIAS:
        out_path = OUTPUT_DIR / f'{name}_num_ques.csv'
        if out_path.exists():
            df = pd.read_csv(out_path)
            total = df['num_questions'].sum()
            n = df['instance_id'].nunique()
            print(f'  {name:<22} {total/n:>8.2f}')
    print(f'  {"[Paper] Qwen 3 Coder":<22} {"6.02":>8}  referência')


if __name__ == '__main__':
    main()
