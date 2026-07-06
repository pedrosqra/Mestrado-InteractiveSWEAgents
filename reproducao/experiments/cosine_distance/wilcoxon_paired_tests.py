"""
wilcoxon_paired_tests.py
--------------------------------
Executa testes estatísticos cientificamente controlados para a comparação
do Qwen2.5 Coder.

Restringe o teste de escala e de simuladores ao subconjunto 100% pareado:
Realiza um "Inner Join" para os testes pareados de Wilcoxon, de modo que
se um modelo não interagiu em uma issue, essa issue é descartada do par,
evitando comparar um valor real de Cosseno com um NaN ou 0.0.

Uso:
    python3 wilcoxon_paired_tests.py
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from pathlib import Path

DIR = Path(__file__).resolve().parent
BATERIAS = [
    'qwen_7b_gpt', 'qwen_7b_gemini',
    'qwen_14b_gpt', 'qwen_14b_gemini',
    'qwen_32b_gpt', 'qwen_32b_gemini'
]

def load_and_pivot():
    data_dict = {}
    for name in BATERIAS:
        path = DIR / f'{name}_embedding_results.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            data_dict[name] = {}
        else:
            agg_df = df.groupby('instance_id')['difference_score'].mean().reset_index()
            data_dict[name] = dict(zip(agg_df['instance_id'], agg_df['difference_score']))
    pivot_df = pd.DataFrame(data_dict)
    return pivot_df

def run_paired_test(df, col1, col2, test_name):
    # Pega a interseção de issues que têm valor nos DOIS cenários
    valid_df = df[[col1, col2]].dropna()
    n_pairs = len(valid_df)
    
    print(f"\n  * {test_name}:")
    print(f"    - N= {n_pairs} issues pareadas (excluídas issues com 0 interação)")
    
    if n_pairs < 5:
        print("    - ERRO: Amostra muito pequena para Wilcoxon pareado.")
        return
        
    diff_mean = valid_df[col2].mean() - valid_df[col1].mean()
    stat, p_w = wilcoxon(valid_df[col1], valid_df[col2])
    sig = "SIM" if p_w < 0.05 else "NÃO"
    
    print(f"    - Diferença das médias ({col2} - {col1}): {diff_mean:+.4f}")
    print(f"    - p-valor: {p_w:.4e} | Significativo (alpha=0.05)? {sig}")

def run_analysis():
    print(f"\n============================================================")
    print(f"ANÁLISE ESTATÍSTICA RIGOROSA (Com interseção pareada sem NaN)")
    print(f"============================================================")
    
    pivot_df = load_and_pivot()
    
    # 1. Estatísticas Descritivas
    print("\n### 1. Estatísticas Descritivas (Considerando apenas os turnos em que houve interação)")
    desc = pivot_df.describe().loc[['count', 'mean', 'std', 'min', 'max']]
    print(desc.to_string())
    
    # 2. Comparação de Escala Controlada: 7B vs 14B (Wilcoxon sem correção)
    print("\n### 2. Comparação Controlada de Escala (7B vs 14B - MLX 4-bit local)")
    run_paired_test(pivot_df, 'qwen_7b_gpt', 'qwen_14b_gpt', 'Sob GPT (7B vs 14B)')
    run_paired_test(pivot_df, 'qwen_7b_gemini', 'qwen_14b_gemini', 'Sob Gemini (7B vs 14B)')
    
    # 3. Comparação Controlada de Simulador (GPT vs Gemini)
    print("\n### 3. Comparação de Simuladores (GPT vs Gemini - Wilcoxon pareado)")
    for size in ['7b', '14b', '32b']:
        gpt_col = f'qwen_{size}_gpt'
        gem_col = f'qwen_{size}_gemini'
        run_paired_test(pivot_df, gpt_col, gem_col, f'Qwen2.5 {size.upper()} (GPT vs Gemini)')

if __name__ == '__main__':
    run_analysis()
