import os
"""
llm_judge_swap.py
-----------------
Este script executa um experimento cientificamente controlado para isolar
e quantificar o viés de avaliação do LLM Judge (GPT-4o).

Para cada issue avaliada e cada modelo de agente (7B, 14B, 32B), extraímos:
  - Q_gpt, A_gpt: Pergunta e resposta do primeiro turno sob o simulador GPT-mini
  - Q_gemini, A_gemini: Pergunta e resposta do primeiro turno sob o simulador Gemini Flash

Em seguida, construímos quatro logs de Turno 1 e avaliamos com GPT-4o:
  1. Original GPT:      (Q_gpt, A_gpt)
  2. Original Gemini:   (Q_gemini, A_gemini)
  3. Swap GPT-Gemini:   (Q_gpt, A_gemini)   --> Pergunta GPT pareada com Resposta Gemini
  4. Swap Gemini-GPT:   (Q_gemini, A_gpt)   --> Pergunta Gemini pareada com Resposta GPT

Se o juiz avaliar com notas substancialmente maiores o swap GPT-Gemini em relação ao
original GPT (para a mesma pergunta Q_gpt), isso mede o viés direto da resposta do simulador.

Uso:
    python3 llm_judge_swap.py
"""

import json
import re
import time
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import litellm

# ─── Configurações ─────────────────────────────────────────────────────────────
API_KEY = os.environ.get("OPENAI_API_KEY")

BASE_DIR = Path(__file__).parent.parent      # experiments/
QA_DIR = BASE_DIR / 'extracted_qa_pairs'
OUTPUT_DIR = Path(__file__).parent            # llm_as_judge/
CSV_PATH = Path(__file__).resolve().parents[3] / 'data' / 'sample_30_underspecified.csv'

MODELS = ['7b', '14b', '32b']

# ─── Função de Avaliação do Juiz ──────────────────────────────────────────────

def evaluate_answer_with_retry(problem_statement, question, answer, max_retries=5):
    """Avalia a resposta em uma escala de 1 a 5 usando GPT-4o, com retentativas em caso de erro."""
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
    
    for attempt in range(max_retries):
        try:
            response = litellm.completion(
                api_key=API_KEY,
                model="openai/gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant adept at nuanced evaluations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            score_text = response['choices'][0]['message']['content'].strip()
            score_match = re.search(r'<score>(\d+)</score>', score_text)
            if score_match:
                score = int(score_match.group(1))
                if 1 <= score <= 5:
                    return score
            print(f"  [Tentativa {attempt+1}] Score inválido ou não encontrado na resposta: {score_text}")
        except Exception as e:
            print(f"  [Tentativa {attempt+1}] Erro na chamada à API: {e}")
        time.sleep(2 ** attempt)
    return None

# ─── Processamento dos Dados ──────────────────────────────────────────────────

def safe_wilcoxon(x, y):
    if np.array_equal(x, y):
        return 0.0, 1.0
    try:
        stat, p = wilcoxon(x, y)
        return stat, p
    except ValueError as e:
        if "do not work if x - y is zero for all elements" in str(e) or "zero_method" in str(e):
            return 0.0, 1.0
        raise e

def main():
    csv_out = OUTPUT_DIR / 'llm_judge_swap_results.csv'
    
    if csv_out.exists():
        print(f"Arquivo de resultados encontrado: {csv_out}")
        print("⏭ Pulando chamadas de API e carregando resultados salvos no disco...")
        results_df = pd.read_csv(csv_out)
    else:
        print('Carregando problem statements...')
        df = pd.read_csv(CSV_PATH)
        problem_statements = dict(zip(df['instance_id'], df['problem_statement']))
        print(f'  {len(problem_statements)} instâncias carregadas.\n')

        # Carregar dados dos QA pairs
        qa_data = {}
        for size in MODELS:
            for sim in ['gpt', 'gemini']:
                name = f'qwen_{size}_{sim}'
                qa_path = QA_DIR / f'{name}_qa_pairs.json'
                if not qa_path.exists():
                    print(f" Arquivo não encontrado: {qa_path}")
                    return
                with open(qa_path) as f:
                    # Transforma a lista em dicionário {instance_id: qa_pairs}
                    raw_list = json.load(f)
                    qa_data[name] = {item['instance_id']: item['qa_pairs'] for item in raw_list}

        results = []

        print("=" * 70)
        print("INICIANDO EXPERIMENTO DE SWAP (TURNO 1)")
        print("=" * 70)

        for size in MODELS:
            print(f"\nProcessando modelo: Qwen2.5-{size.upper()}")
            paired_count = 0
            
            gpt_battery = f'qwen_{size}_gpt'
            gemini_battery = f'qwen_{size}_gemini'
            
            for instance_id, problem in problem_statements.items():
                gpt_qa = qa_data[gpt_battery].get(instance_id, [])
                gemini_qa = qa_data[gemini_battery].get(instance_id, [])
                
                # Precisamos que ambos tenham interagido no Turno 1
                if len(gpt_qa) > 0 and len(gemini_qa) > 0:
                    q_gpt, a_gpt = gpt_qa[0]
                    q_gemini, a_gemini = gemini_qa[0]
                    
                    paired_count += 1
                    print(f"   [{paired_count}] Avaliando issue {instance_id}...")
                    
                    # Executar as 4 avaliações com GPT-4o
                    s_orig_gpt = evaluate_answer_with_retry(problem, q_gpt, a_gpt)
                    s_orig_gem = evaluate_answer_with_retry(problem, q_gemini, a_gemini)
                    s_swap_gpt_gem = evaluate_answer_with_retry(problem, q_gpt, a_gemini)
                    s_swap_gem_gpt = evaluate_answer_with_retry(problem, q_gemini, a_gpt)
                    
                    if None not in (s_orig_gpt, s_orig_gem, s_swap_gpt_gem, s_swap_gem_gpt):
                        results.append({
                            'model': size,
                            'instance_id': instance_id,
                            'q_gpt': q_gpt,
                            'a_gpt': a_gpt,
                            'q_gemini': q_gemini,
                            'a_gemini': a_gemini,
                            'score_orig_gpt': s_orig_gpt,
                            'score_orig_gemini': s_orig_gem,
                            'score_swap_gpt_gem': s_swap_gpt_gem,
                            'score_swap_gem_gpt': s_swap_gem_gpt
                        })
                    else:
                        print(f"   Falha na obtenção de score para {instance_id}. Pulando.")
                        
            print(f"   Encontradas {paired_count} issues pareadas com interação para o modelo {size.upper()}.")

        # Salvar resultados brutos
        results_df = pd.DataFrame(results)
        results_df.to_csv(csv_out, index=False)
        print(f"\nResultados brutos salvos em {csv_out}")

    # ─── Análise Estatística dos Resultados ───────────────────────────────────
    print("\n" + "=" * 70)
    print("ANÁLISE ESTATÍSTICA: WILCOXON E ANÁLISE DE VIÉS")
    print("=" * 70)

    for size in MODELS:
        sub_df = results_df[results_df['model'] == size]
        if sub_df.empty:
            continue
            
        n = len(sub_df)
        print(f"\n--- Modelo Qwen2.5-{size.upper()} (N = {n} issues pareadas) ---")
        
        # Médias descritivas
        mean_orig_gpt = sub_df['score_orig_gpt'].mean()
        mean_orig_gem = sub_df['score_orig_gemini'].mean()
        mean_swap_gpt_gem = sub_df['score_swap_gpt_gem'].mean()
        mean_swap_gem_gpt = sub_df['score_swap_gem_gpt'].mean()
        
        print(f"  Médias dos scores originais (Turno 1):")
        print(f"    - Original GPT-mini (Q_gpt, A_gpt):       {mean_orig_gpt:.3f}")
        print(f"    - Original Gemini Flash (Q_gem, A_gem):   {mean_orig_gem:.3f}")
        print(f"    Delta Original (Gemini - GPT):           {mean_orig_gem - mean_orig_gpt:+.3f}")
        
        print(f"  Médias dos scores com Swap de Respostas:")
        print(f"    - Pergunta GPT + Resposta Gemini (Q_gpt, A_gem): {mean_swap_gpt_gem:.3f}")
        print(f"    - Pergunta Gemini + Resposta GPT (Q_gem, A_gpt): {mean_swap_gem_gpt:.3f}")
        
        # Teste 1: Viés do Simulador sobre Pergunta GPT (Q_gpt + A_gpt vs Q_gpt + A_gemini)
        stat_bias1, p_bias1 = safe_wilcoxon(sub_df['score_orig_gpt'], sub_df['score_swap_gpt_gem'])
        print(f"\n  [Teste 1] Viés sobre Pergunta GPT (Q_gpt + A_gpt vs Q_gpt + A_gemini):")
        print(f"    - Efeito de trocar resposta de GPT por Gemini: {mean_swap_gpt_gem - mean_orig_gpt:+.3f}")
        print(f"    - Wilcoxon p-value: {p_bias1:.2e} | Significativo? {'SIM' if p_bias1 < 0.05 else 'NÃO'}")

        # Teste 2: Viés do Simulador sobre Pergunta Gemini (Q_gemini + A_gpt vs Q_gemini + A_gemini)
        stat_bias2, p_bias2 = safe_wilcoxon(sub_df['score_swap_gem_gpt'], sub_df['score_orig_gemini'])
        print(f"  [Teste 2] Viés sobre Pergunta Gemini (Q_gemini + A_gpt vs Q_gemini + A_gemini):")
        print(f"    - Efeito de trocar resposta de GPT por Gemini: {mean_orig_gem - mean_swap_gem_gpt:+.3f}")
        print(f"    - Wilcoxon p-value: {p_bias2:.2e} | Significativo? {'SIM' if p_bias2 < 0.05 else 'NÃO'}")

        # Teste 3: Diferença real da Pergunta sob simulador GPT (Q_gpt + A_gpt vs Q_gemini + A_gpt)
        stat_q1, p_q1 = safe_wilcoxon(sub_df['score_orig_gpt'], sub_df['score_swap_gem_gpt'])
        print(f"\n  [Teste 3] Qualidade da Pergunta controlando resposta como GPT (Q_gpt + A_gpt vs Q_gemini + A_gpt):")
        print(f"    - Diferença (Q_gemini - Q_gpt): {mean_swap_gem_gpt - mean_orig_gpt:+.3f}")
        print(f"    - Wilcoxon p-value: {p_q1:.2e} | Significativo? {'SIM' if p_q1 < 0.05 else 'NÃO'}")

        # Teste 4: Diferença real da Pergunta sob simulador Gemini (Q_gpt + A_gemini vs Q_gemini + A_gemini)
        stat_q2, p_q2 = safe_wilcoxon(sub_df['score_swap_gpt_gem'], sub_df['score_orig_gemini'])
        print(f"  [Teste 4] Qualidade da Pergunta controlando resposta como Gemini (Q_gpt + A_gemini vs Q_gemini + A_gemini):")
        print(f"    - Diferença (Q_gemini - Q_gpt): {mean_orig_gem - mean_swap_gpt_gem:+.3f}")
        print(f"    - Wilcoxon p-value: {p_q2:.2e} | Significativo? {'SIM' if p_q2 < 0.05 else 'NÃO'}")

    print("\n" + "=" * 70)
    print("Concluído com Sucesso!")
    print("=" * 70)


if __name__ == '__main__':
    main()
