"""
analyze_questions_responses.py
------------------------------
Analisa se o simulador responde de forma diferente a perguntas do 7B vs 14B no primeiro turno,
para verificar se a melhora na escala é sinal real (perguntas melhores forçando respostas melhores)
ou se o simulador responde com o mesmo teor e a diferença nas médias gerais se dá pelo volume de ruído.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).parent.parent
QA_DIR = BASE_DIR / 'extracted_qa_pairs'
SWAP_CSV = BASE_DIR / 'llm_as_judge' / 'llm_judge_swap_results.csv'

def main():
    print("Analisando respostas do simulador no Turno 1 para 7B, 14B, 32B...")
    
    # 1. Carregar os dados de swap do Turno 1
    if not SWAP_CSV.exists():
        print(f"Arquivo {SWAP_CSV} não encontrado.")
        return
        
    df = pd.read_csv(SWAP_CSV)
    
    print("\n--- Estatísticas de Verbosidade do Turno 1 ---")
    for size in ['7b', '14b', '32b']:
        sub = df[df['model'] == size]
        n = len(sub)
        
        # Comprimento das perguntas e respostas
        len_q_gpt = sub['q_gpt'].str.len().mean()
        len_a_gpt = sub['a_gpt'].str.len().mean()
        len_q_gem = sub['q_gemini'].str.len().mean()
        len_a_gem = sub['a_gemini'].str.len().mean()
        
        # DEFINIÇÃO OPERACIONAL DE RECUSA (definição operacional de recusa)
        # Optou-se por uma definição conservadora baseada exclusivamente na frase
        # canônica de recusa especificada no prompt do simulador. Critérios mais
        # abrangentes ('unrelated', 'unsure') foram considerados inicialmente, mas
        # inspeção manual dos 9 casos flagrados revelou que todos eram falsos
        # positivos (ocorrência das palavras em contexto de código Django ou em
        # respostas substantivas com admissão parcial de incerteza). A definição
        # conservadora privilegia alta precisão na identificação de recusas
        # explícitas, podendo não capturar formulações alternativas semanticamente
        # equivalentes — subestimação discutida nos threats to validity.
        #
        # Síntese dos achados:
        # "A ausência de informação útil foi operacionalizada como a ocorrência da
        # frase canônica 'I don't have that information', explicitamente especificada
        # no prompt do simulador. Critérios baseados em palavras-chave adicionais
        # foram descartados após inspeção manual revelar falsos positivos sistemáticos.
        # Adotou-se, portanto, uma definição conservadora para garantir validade
        # interna, com possível subestimação de recusas expressas por outras
        # formulações."
        no_info_gpt = sub['a_gpt'].str.contains("I don't have that information", case=False).mean() * 100
        no_info_gem = sub['a_gemini'].str.contains("I don't have that information", case=False).mean() * 100
        
        print(f"\nModelo Qwen2.5-{size.upper()} (N = {n} issues):")
        print(f"  Sob simulador GPT-mini:")
        print(f"    - Comprimento médio da pergunta do agente: {len_q_gpt:.1f} caracteres")
        print(f"    - Comprimento médio da resposta do proxy:  {len_a_gpt:.1f} caracteres")
        print(f"    - Respostas do proxy sem informação útil: {no_info_gpt:.1f}%")
        print(f"  Sob simulador Gemini Flash:")
        print(f"    - Comprimento médio da pergunta do agente: {len_q_gem:.1f} caracteres")
        print(f"    - Comprimento médio da resposta do proxy:  {len_a_gem:.1f} caracteres")
        print(f"    - Respostas do proxy sem informação útil: {no_info_gem:.1f}%")

        # AMBIGUIDADE CAUSAL DA DIFERENÇA ENTRE SIMULADORES (Seção de Limitações)
        # A diferença na taxa de fallback entre GPT-4o-mini e Gemini Flash não permite
        # inferir causalidade. O dado comportamental é compatível com três hipóteses:
        #
        #   1. GPT-4o-mini é mais restritivo (design): segue a instrução do prompt com
        #      mais fidelidade e rejeita perguntas técnicas detalhadas como instruído.
        #
        #   2. GPT-4o-mini é menos capaz (conhecimento): genuinamente não tem cobertura
        #      factual suficiente para responder as perguntas cirúrgicas do 14B.
        #
        #   3. Gemini Flash confabula (risco de validade interna): taxa de rejeição
        #      próxima a zero pode refletir propensão a gerar respostas plausíveis
        #      mesmo sem informação de base — o que inflaria os scores do juiz sob
        #      esse simulador e comprometeria a comparação entre baterias.
        #
        # Nota: esta ambiguidade diz respeito à *interpretação* da diferença entre
        # simuladores, não à validade da métrica em si (tratada no bloco acima).
        # Deve aparecer nos threats to validity, não na seção de Método.

    # 2. Verificar a distribuição de notas originais no Turno 1
    print("\n--- Médias dos Scores Originais no Turno 1 (GPT-4o Judge) ---")
    for size in ['7b', '14b', '32b']:
        sub = df[df['model'] == size]
        print(f"Qwen2.5-{size.upper()}:")
        print(f"  - Turno 1 sob GPT (Original):    {sub['score_orig_gpt'].mean():.3f}")
        print(f"  - Turno 1 sob Gemini (Original): {sub['score_orig_gemini'].mean():.3f}")

if __name__ == '__main__':
    main()
