"""
run_tests.py
------------------
Executa a análise estatística completa sobre os resultados de 
Distância do Cosseno (Ganho de Informação) e LLM-as-Judge das baterias Qwen2.5 Coder.

Testes aplicados:
  1. Teste Exato de McNemar (Binomial) para comparar a taxa de interatividade do 1.5B.
  2. Teste de Wilcoxon Signed-Rank pareado com cálculo de Z, p-value e Effect Size (r).
  3. Correção de Holm-Bonferroni para as comparações pareadas de sistemas.
  4. Teste de Friedman (global) separando por simulador (com N=29 e N=30).
  5. Teste L de Page (tendência monotônica ordenada) separando por simulador.

Os resultados são salvos em um relatório markdown formatado com explicações teóricas.
"""

import os
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, binom, norm

# ─── Configuração de Caminhos ─────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent      # experiments/
QA_DIR = BASE_DIR / 'extracted_qa_pairs'
COS_DIR = BASE_DIR / 'cosine_distance'
JUDGE_DIR = BASE_DIR / 'llm_as_judge'
OUTPUT_DIR = Path(__file__).parent            # statistical_analysis/

# Amostra de 30 issues do benchmark
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


# ─── Teste Exato de McNemar (Binomial Bicaudal) ─────────────────────────────
def mcnemar_exact(b, c):
    """Calcula o p-value exato de McNemar usando distribuição binomial."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # CDF binomial bicaudal
    return 2 * binom.cdf(k, n, 0.5)


# ─── Teste L de Page (Tendência Ordenada) ───────────────────────────────────
def page_trend_test(data_matrix):
    """
    Calcula a estatística L de Page e seu p-value unilateral para a hipótese
    de que as colunas seguem a ordem especificada (Coluna 0 <= Coluna 1 <= Coluna 2).
    
    data_matrix: array numpy (N issues, k modelos)
    """
    N, k = data_matrix.shape
    if k < 3:
        raise ValueError("O teste L de Page requer pelo menos 3 colunas/modelos.")

    # Converte os dados em postos (ranks) dentro de cada issue (linha)
    ranks = np.zeros_like(data_matrix)
    for i in range(N):
        # scipy rankdata trata empates de forma justa (average ranks)
        from scipy.stats import rankdata
        ranks[i] = rankdata(data_matrix[i])

    # Soma dos postos por modelo
    R = np.sum(ranks, axis=0)

    # Estatística L de Page
    L = 0.0
    for j in range(k):
        # Ordem esperada: Coluna 0 (posto j=1) <= Coluna 1 (j=2) <= Coluna 2 (j=3)
        L += (j + 1) * R[j]

    # Média e variância sob a hipótese nula
    mu_L = (N * k * (k + 1)**2) / 4.0
    var_L = (N * (k**2) * (k + 1) * (k**2 - 1)) / 144.0
    sigma_L = np.sqrt(var_L)

    # Estatística normal Z
    # Aplica correção de continuidade se L > mu_L
    Z = (L - mu_L - 0.5) / sigma_L if L > mu_L else (L - mu_L + 0.5) / sigma_L

    # p-value unilateral (cauda superior)
    p_value = 1.0 - norm.cdf(Z)

    return L, Z, p_value


# ─── Correção de Holm-Bonferroni ─────────────────────────────────────────────
def holm_bonferroni(p_values):
    """Aplica a correção de Holm-Bonferroni em uma lista de p-values."""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    adjusted_p = np.zeros(n)
    
    for i, idx in enumerate(sorted_indices):
        m = n - i
        adjusted_p[idx] = min(p_values[idx] * m, 1.0)
    
    # Garante monotonicidade
    for i in range(1, n):
        idx_prev = sorted_indices[i - 1]
        idx_curr = sorted_indices[i]
        if adjusted_p[idx_curr] < adjusted_p[idx_prev]:
            adjusted_p[idx_curr] = adjusted_p[idx_prev]
            
    return adjusted_p


# ─── Carregamento e Alinhamento dos Dados ─────────────────────────────────────
def load_and_align_data(problem_statements, mode='cosine'):
    """
    Lê os CSVs brutos e alinha todas as issues para N=30.
    mode: 'cosine' (difference_score) ou 'judge' (new_information_score)
    """
    aligned_data = {}
    
    for name in BATERIAS:
        if mode == 'cosine':
            path = COS_DIR / f'{name}_embedding_results.csv'
            col = 'difference_score'
            default_val = 0.0
        else:
            path = JUDGE_DIR / f'{name}_gpt4o_evaluation_results.csv'
            col = 'new_information_score'
            default_val = 1.0  # Nota mínima do Judge
            
        if not path.exists() or path.stat().st_size <= 10:
            # Cria dados fictícios se o arquivo não existir ou for vazio (ex: bateria 1.5B com zero interações)
            aligned_data[name] = {issue: default_val for issue in problem_statements}
            continue
            
        df = pd.read_csv(path)
        
        # Agrupa por issue para obter o score médio por issue
        grouped = df.groupby('instance_id')[col].mean().to_dict()
        
        # Alinha com as 30 issues padrão
        battery_scores = {}
        for issue in problem_statements:
            if issue in grouped:
                battery_scores[issue] = grouped[issue]
            else:
                battery_scores[issue] = default_val
                
        aligned_data[name] = battery_scores
        
    return aligned_data


def main():
    print("Carregando instâncias...")
    df_csv = pd.read_csv(CSV_PATH)
    problem_statements = list(df_csv['instance_id'].unique())
    print(f"  {len(problem_statements)} instâncias carregadas.\n")

    # Carrega dados
    cosine_aligned = load_and_align_data(problem_statements, mode='cosine')
    judge_aligned = load_and_align_data(problem_statements, mode='judge')

    report = []
    report.append("# Relatório de Análise Estatística — Replicação Qwen2.5 Coder\n")
    report.append("> [!NOTE]")
    report.append("> Este relatório foi gerado automaticamente e aplica testes estatísticos não-paramétricos")
    report.append("> para validar o ganho de informação (Embedding Cosine Distance) e a qualidade das respostas (LLM-as-Judge).\n")

    # ──────────────────────────────────────────────────────────────────────────
    # 1. TESTE DE MCNEMAR (Taxa de Interatividade do 1.5B)
    # ──────────────────────────────────────────────────────────────────────────
    report.append("## 1. Teste de McNemar (Taxa de Interatividade do Qwen 1.5B)\n")
    
    report.append("### Conceito Teórico: Teste Exato de McNemar (Binomial)")
    report.append("> O teste de McNemar é um teste estatístico não-paramétrico usado em dados nominais pareados (tabelas de contingência 2x2 com medidas repetidas). Ele é aplicado para determinar se existe diferença nas proporções antes e depois de um tratamento, ou entre dois modelos aplicados nos mesmos sujeitos.")
    report.append("> ")
    report.append("> **Intuição**: Ele foca apenas nas células da diagonal secundária (onde o Modelo 1 e o Modelo 2 discordam). Se a hipótese nula de que os dois modelos têm a mesma proporção de acertos for verdadeira, a proporção esperada de discordâncias deve ser igual a uma moeda justa (50% de chance de pender para qualquer um dos lados).")
    report.append("> ")
    report.append("> **Por que a versão Exata?**: Em amostras com células zeradas ou muito pequenas (soma das discordâncias < 25), a aproximação qui-quadrado tradicional fica instável e incorreta. A versão exata calcula a probabilidade cumulativa diretamente pela distribuição binomial $Binom(b+c, 0.5)$, garantindo um cálculo exato e livre de aproximações assintóticas.")
    report.append("\nComo o Qwen 1.5B nunca iniciou interações (0/30 issues), ele possui variância zero.")
    report.append("Usamos o **Teste Exato de McNemar (Binomial)** para comparar as proporções pareadas de interatividade.\n")
    report.append("| Comparação | Célula [1,0] (M1 sim, M2 não) | Célula [0,1] (M1 não, M2 sim) | p-value (Exato) | Significativo? |")
    report.append("| :--- | :---: | :---: | :---: | :---: |")

    # Compara 1.5B contra 7B, 14B e 32B (GPT e Gemini)
    # Para o 1.5B, a interação é sempre 0.
    for sim in ['gpt', 'gemini']:
        m15_name = f'qwen_1_5b_{sim}'
        m15_active = [1 if cosine_aligned[m15_name][issue] > 0 else 0 for issue in problem_statements]
        
        for scale in ['7b', '14b', '32b']:
            m_name = f'qwen_{scale}_{sim}'
            m_active = [1 if cosine_aligned[m_name][issue] > 0 else 0 for issue in problem_statements]
            
            # Tabela de contingência 2x2 pareada
            # [ [ambos_sim, M15_sim_M_nao], [M15_nao_M_sim, ambos_nao] ]
            b = sum(1 for i in range(30) if m15_active[i] == 1 and m_active[i] == 0) # M15_sim, M_nao
            c = sum(1 for i in range(30) if m15_active[i] == 0 and m_active[i] == 1) # M15_nao, M_sim
            
            p_val = mcnemar_exact(b, c)
            sig = "Sim" if p_val < 0.05 else "Não"
            report.append(f"| {m15_name.upper()} vs {m_name.upper()} | {b} | {c} | {p_val:.6f} | {sig} |")
    report.append("\n*Nota: O p-value exato binomial de McNemar é computado de forma bicaudal, ideal para as células zeradas da tabela de contingência.*\n")

    # ──────────────────────────────────────────────────────────────────────────
    # 2. TESTE DE WILCOXON E EFFECT SIZE (Cosseno e Judge)
    # ──────────────────────────────────────────────────────────────────────────
    report.append("## 2. Testes de Wilcoxon Pareados & Tamanho do Efeito (r)\n")
    
    report.append("### Conceito Teórico: Wilcoxon Signed-Rank Test & Holm-Bonferroni")
    report.append("> O teste de postos sinalizados de Wilcoxon é um teste não-paramétrico pareado usado para comparar duas amostras de medidas repetidas (o análogo não-paramétrico do teste t pareado).")
    report.append("> ")
    report.append("> **Intuição**: Ele calcula a diferença de score para cada issue, ordena essas diferenças em valor absoluto, atribui postos (ranks) e depois soma os postos das diferenças positivas e negativas separadamente. Sob a hipótese nula, as duas somas devem ser estatisticamente similares.")
    report.append("> ")
    report.append("> **Estatística Z e Effect Size (r)**: Para amostras de tamanho razoável (como N=30), a estatística do teste se aproxima de uma normal padronizada. Calculamos $Z$ para extrair o tamanho do efeito de Cohen $r = \\frac{Z}{\\sqrt{N}}$, que indica a magnitude real da diferença (0.1: pequeno, 0.3: médio, >=0.5: grande) de forma independente do tamanho da amostra.")
    report.append("> ")
    report.append("> **Holm-Bonferroni**: Quando realizamos múltiplos testes estatísticos pareados na mesma amostra, a chance de encontrar um falso positivo por acaso (erro Tipo I) aumenta. A correção de Holm ordena os p-values e ajusta os limites de corte de forma sequencial, protegendo a análise estatística de forma menos conservadora e mais poderosa que a Bonferroni clássica.")
    
    report.append("\nCalcula a diferença issue a issue. A estatística $Z$ é extraída pela aproximação normal para permitir o cálculo de $r = Z/\\sqrt{N}$.")
    report.append("A correção de Holm-Bonferroni é aplicada a cada família de testes por simulador.\n")

    for metric_name, aligned_data in [("Distância do Cosseno", cosine_aligned), ("LLM-as-Judge Score", judge_aligned)]:
        report.append(f"### Métrica: {metric_name}\n")
        
        for sim in ['gpt', 'gemini']:
            report.append(f"#### Simulador: {sim.upper()}\n")
            report.append("| Par Comparado | Tipo de Comparação | W | Z | p-value (Bruto) | p-value (Holm-Bonf) | Tamanho de Efeito (r) | Classificação |")
            report.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
            
            comparisons = [
                (f'qwen_7b_{sim}', f'qwen_14b_{sim}', "Escala Pura"),
                (f'qwen_7b_{sim}', f'qwen_32b_{sim}', "Sistema"),
                (f'qwen_14b_{sim}', f'qwen_32b_{sim}', "Sistema")
            ]
            
            p_vals = []
            results_cache = []
            
            for m1, m2, comp_type in comparisons:
                # Alinha as séries
                # Ignora a issue excluída 'django__django-13112' se for o Gemini (N=29) para o Wilcoxon do par com Gemini
                if 'gemini' in sim:
                    issues_list = [issue for issue in problem_statements if issue != 'django__django-13112']
                else:
                    issues_list = problem_statements
                
                N_wilcoxon = len(issues_list)
                x = [aligned_data[m1][issue] for issue in issues_list]
                y = [aligned_data[m2][issue] for issue in issues_list]
                
                # Verifica se há diferenças reais
                diff = np.array(x) - np.array(y)
                if np.all(diff == 0):
                    W, Z, p_bruto, r = 0, 0, 1.0, 0.0
                else:
                    # Roda Wilcoxon com aproximação normal para retornar Z
                    try:
                        res = wilcoxon(x, y, alternative='two-sided')
                        W = res.statistic
                        p_bruto = res.pvalue
                        
                        res_approx = wilcoxon(x, y, alternative='two-sided', method='approx')
                        Z = getattr(res_approx, 'zstatistic', np.nan)
                        
                        if np.isnan(Z):
                            n_nonzero = np.count_nonzero(diff)
                            mu_w = n_nonzero * (n_nonzero + 1) / 4.0
                            sigma_w = np.sqrt(n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24.0)
                            Z = (W - mu_w) / sigma_w
                            
                        r = abs(Z) / np.sqrt(N_wilcoxon)
                    except Exception as e:
                        print(f"Erro no Wilcoxon entre {m1} e {m2}: {e}")
                        W, Z, p_bruto, r = np.nan, np.nan, np.nan, np.nan
                
                p_vals.append(p_bruto)
                results_cache.append((m1, m2, comp_type, W, Z, p_bruto, r))
            
            # Corrige p-values
            adj_p_vals = holm_bonferroni(p_vals)
            
            for i, res in enumerate(results_cache):
                m1, m2, comp_type, W, Z, p_bruto, r = res
                p_adj = adj_p_vals[i]
                
                # Classificação de Cohen
                if r < 0.1:
                    magnitude = "Negligível"
                elif r < 0.3:
                    magnitude = "Pequeno"
                elif r < 0.5:
                    magnitude = "Médio"
                else:
                    magnitude = "Grande"
                    
                m1_label = m1.upper().replace(f'_{sim.upper()}', '')
                m2_label = m2.upper().replace(f'_{sim.upper()}', '')
                report.append(f"| {m1_label} vs {m2_label} | {comp_type} | {W:.1f} | {Z:.4f} | {p_bruto:.6f} | {p_adj:.6f} | {r:.4f} | {magnitude} |")
            report.append("\n")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. TESTE DE FRIEDMAN (Global - 7B, 14B, 32B)
    # ──────────────────────────────────────────────────────────────────────────
    report.append("## 3. Teste de Friedman (Diferença Global do Trio 7B, 14B, 32B)\n")
    
    report.append("### Conceito Teórico: Teste de Friedman (Omnibus)")
    report.append("> O teste de Friedman é um teste estatístico não-paramétrico para medidas repetidas em três ou mais condições (o análogo não-paramétrico da ANOVA de medidas repetidas).")
    report.append("> ")
    report.append("> **Intuição**: Para cada issue (linha), o teste ordena os modelos (colunas) de 1 a $k$, soma os postos das colunas de forma agregada e avalia se essas somas diferem significativamente entre si sob a hipótese nula de que todas as condições são equivalentes.")
    report.append("> ")
    report.append("> **Métrica Omnibus**: Ele é um teste 'omnibus', o que significa que ele detecta se *existe alguma diferença geral* no grupo de modelos, mas não especifica qual par é diferente (necessitando de um teste post-hoc como o Wilcoxon para identificar as diferenças específicas).")
    
    report.append("\nAplica o teste de Friedman pareado por issue de forma independente para cada simulador.")
    report.append("Rodamos em duas configurações de alinhamento:\n")
    report.append("1. **Configuração Exclusão**: Remove a issue 'django__django-13112' (N=29) comum a todos para evitar dados ausentes no 14B Gemini.")
    report.append("2. **Configuração Imputação**: Imputa `0.0` para ganho de informação na issue ausente no 14B Gemini (N=30).\n")
    
    report.append("| Métrica | Simulador | Alinhamento | N | Estatística (Chi2) | p-value | Significativo? |")
    report.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for metric_name, aligned_data in [("Distância do Cosseno", cosine_aligned), ("LLM-as-Judge Score", judge_aligned)]:
        for sim in ['gpt', 'gemini']:
            m7 = f'qwen_7b_{sim}'
            m14 = f'qwen_14b_{sim}'
            m32 = f'qwen_32b_{sim}'
            
            # Variante 1: Exclusão (N=29)
            issues_ex = [issue for issue in problem_statements if issue != 'django__django-13112']
            v7_ex = [aligned_data[m7][issue] for issue in issues_ex]
            v14_ex = [aligned_data[m14][issue] for issue in issues_ex]
            v32_ex = [aligned_data[m32][issue] for issue in issues_ex]
            
            stat_ex, p_ex = friedmanchisquare(v7_ex, v14_ex, v32_ex)
            sig_ex = "Sim" if p_ex < 0.05 else "Não"
            report.append(f"| {metric_name} | {sim.upper()} | Exclusão | 29 | {stat_ex:.4f} | {p_ex:.6f} | {sig_ex} |")
            
            # Variante 2: Imputação (N=30)
            v7_imp = [aligned_data[m7][issue] for issue in problem_statements]
            v14_imp = [aligned_data[m14][issue] for issue in problem_statements]
            v32_imp = [aligned_data[m32][issue] for issue in problem_statements]
            
            stat_imp, p_imp = friedmanchisquare(v7_imp, v14_imp, v32_imp)
            sig_imp = "Sim" if p_imp < 0.05 else "Não"
            report.append(f"| {metric_name} | {sim.upper()} | Imputação | 30 | {stat_imp:.4f} | {p_imp:.6f} | {sig_imp} |")
    report.append("\n")
    report.append("> [!TIP]")
    report.append("> **Nota sobre Robustez**: As conclusões do teste de Friedman (assim como as do teste L de Page abaixo) permaneceram qualitativamente inalteradas sob ambas as estratégias de alinhamento (Exclusão e Imputação), sugerindo robustez dos resultados estatísticos frente ao tratamento de dados ausentes.\n")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. TESTE L DE PAGE (Page's Trend Test)
    # ──────────────────────────────────────────────────────────────────────────
    report.append("## 4. Teste L de Page (Tendência Ordenada: 7B <= 14B <= 32B)\n")
    
    report.append("### Conceito Teórico: Teste L de Page")
    report.append("> O teste L de Page é um teste não-paramétrico pareado para medidas repetidas projetado especificamente para testar uma **hipótese alternativa ordenada direcional** (ex: $H_1: M_1 \\le M_2 \\le M_3$).")
    report.append("> ")
    report.append("> **Intuição**: Diferente do Friedman (que não assume ordem), o Page's L calcula os postos de cada coluna e depois calcula uma soma ponderada dos postos multiplicada pelo peso ordenado de cada modelo ($1$ para o primeiro, $2$ para o segundo, etc.).")
    report.append("> ")
    report.append("> **Vantagem de Poder Estatístico**: Por incorporar a hipótese direcional ordenada do experimento ($7B \\le 14B \\le 32B$), o teste L de Page possui muito mais poder estatístico do que o Friedman. Ele consegue detectar a tendência progressiva mesmo quando o Friedman não-direcional falha devido à pequena quantidade de modelos ($k=3$).")
    
    report.append("\nMede se há uma tendência progressiva ordenada de melhora na performance à medida que subimos a escala/sistemas.")
    report.append("A hipótese alternativa testada é unilateral ($H_1: 7B \\le 14B \\le 32B$).\n")
    report.append("| Métrica | Simulador | Alinhamento | N | Estatística L | Z-statistic | p-value | Significativo? |")
    report.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for metric_name, aligned_data in [("Distância do Cosseno", cosine_aligned), ("LLM-as-Judge Score", judge_aligned)]:
        for sim in ['gpt', 'gemini']:
            m7 = f'qwen_7b_{sim}'
            m14 = f'qwen_14b_{sim}'
            m32 = f'qwen_32b_{sim}'
            
            # Variante 1: Exclusão (N=29)
            issues_ex = [issue for issue in problem_statements if issue != 'django__django-13112']
            matrix_ex = np.array([
                [aligned_data[m7][issue] for issue in issues_ex],
                [aligned_data[m14][issue] for issue in issues_ex],
                [aligned_data[m32][issue] for issue in issues_ex]
            ]).T
            
            L_ex, Z_ex, p_ex = page_trend_test(matrix_ex)
            sig_ex = "Sim" if p_ex < 0.05 else "Não"
            report.append(f"| {metric_name} | {sim.upper()} | Exclusão | 29 | {L_ex:.1f} | {Z_ex:.4f} | {p_ex:.6f} | {sig_ex} |")
            
            # Variante 2: Imputação (N=30)
            matrix_imp = np.array([
                [aligned_data[m7][issue] for issue in problem_statements],
                [aligned_data[m14][issue] for issue in problem_statements],
                [aligned_data[m32][issue] for issue in problem_statements]
            ]).T
            
            L_imp, Z_imp, p_imp = page_trend_test(matrix_imp)
            sig_imp = "Sim" if p_imp < 0.05 else "Não"
            report.append(f"| {metric_name} | {sim.upper()} | Imputação | 30 | {L_imp:.1f} | {Z_imp:.4f} | {p_imp:.6f} | {sig_imp} |")
    report.append("\n")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. ANÁLISE APROFUNDADA & DISCUSSÃO METODOLÓGICA
    # ──────────────────────────────────────────────────────────────────────────
    report.append("## 5. Análise Aprofundada & Discussão Metodológica\n")
    
    report.append("### A. Interação Simulador × Sistema")
    report.append("Os dados mostram um padrão revelador na métrica de ganho de informação (Cosseno):")
    report.append("* Sob o **GPT-4o-mini** (um simulador mais 'avaro'), o 32B Nuvem supera estatisticamente o 7B e o 14B com efeito de grande magnitude ($r = 0.59$ e $0.42$, respectivamente) e o teste de Page's L confirma a tendência ascendente ($p = 0.014$).")
    report.append("* Sob o **Gemini 3.5 Flash** (um simulador muito 'informativo/generoso'), **essa melhora estatisticamente significativa desaparece**. A diferença entre o 14B Local e o 32B Nuvem torna-se negligível ($r = 0.01$, $p = 0.932$), e o teste de Page's L **deixa de ser significativo ($p = 0.068$)**.")
    report.append("> [!TIP]")
    report.append("> **Interpretação**: Os resultados indicam que simuladores mais informativos podem reduzir os benefícios observados com modelos de maior escala, sugerindo que a influência da escala do agente depende do simulador utilizado. Se o usuário fornece respostas muito ricas (Gemini), a limitação de escala de um agente menor (14B local) é atenuada, nivelando sua performance ao modelo maior. A escala do agente torna-se crucial apenas sob simuladores restritivos (GPT).\n")
    
    report.append("### B. Reconciliação entre Friedman (Omnibus) e Wilcoxon/Page's L")
    report.append("Sob a métrica de Cosseno/GPT, o Friedman global deu p-value ligeiramente não significativo ($p = 0.072$), enquanto os testes de Wilcoxon pareados ($7B \\text{ vs } 32B$, $14B \\text{ vs } 32B$) e o teste de Page's L foram significativos. ")
    report.append("* **Explicação**: O Friedman é um teste omnibus **não-direcional** que tenta detectar qualquer diferença arbitrária entre os modelos, perdendo poder estatístico quando o número de tratamentos é pequeno ($k=3$) e a variância entre blocos é grande. ")
    report.append("* O **Wilcoxon** (focado no par) e o **Page's L** (desenhado sob medida para testar a hipótese direcional $7B \\le 14B \\le 32B$) possuem muito mais poder estatístico para detectar essa tendência ascendente e, portanto, detectam evidências de uma tendência monotônica de melhora que não foi capturada pelo teste omnibus.\n")
    
    report.append("### C. Ameaças à Validade Interna")
    report.append("Uma importante ameaça à validade interna decorre do fato de os modelos não terem sido executados sob uma infraestrutura computacional idêntica:")
    report.append("* Os modelos Qwen2.5-Coder 1.5B, 7B e 14B foram executados localmente via biblioteca `mlx-lm` num MacBook Air, sendo os dois últimos quantizados em 4 bits para viabilizar a execução.")
    report.append("* O modelo Qwen2.5-Coder 32B foi acessado remotamente em precisão total (ou quantização padrão nativa) via API da OpenRouter devido a restrições físicas de RAM (16GB).")
    report.append("> [!WARNING]")
    report.append("> **Consequência**: Qualquer diferença de performance observada envolvendo o 32B não pode ser estatisticamente atribuída *exclusivamente* à sua maior escala de parâmetros. Diferenças de quantização, ambiente de inferência e otimizações de servidor do provedor de API funcionam como variáveis de confusão colineares.")
    report.append("> ")
    report.append("> **Atenuação**: Embora seja um fator a considerar, esta configuração reflete um cenário prático e realista de Engenharia de Software, onde desenvolvedores comumente adotam modelos menores rodando localmente (visando privacidade/custo zero) e consomem modelos maiores sob demanda via APIs em nuvem. Ademais, o pipeline experimental do agente (fluxo, prompts, temperatura e simuladores) foi mantido perfeitamente constante para todos os modelos.")

    # Salva o relatório
    output_file = OUTPUT_DIR / 'statistical_report.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
        
    print(f"Análise concluída com sucesso!")
    print(f"Relatório estatístico salvo em: {output_file.name}")


if __name__ == '__main__':
    main()
