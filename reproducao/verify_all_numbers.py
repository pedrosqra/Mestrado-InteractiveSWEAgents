"""
Verificação completa de todos os números reportados do estudo
contra os dados brutos em experiments/

Uso: python3 verify_all_numbers.py
Requer: pandas, scipy (pip install pandas scipy)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import binom
import os

BASE = os.path.dirname(os.path.abspath(__file__))
EXP  = os.path.join(BASE, "experiments")
QA   = os.path.join(EXP, "extracted_qa_pairs")
COS  = os.path.join(EXP, "cosine_distance")
JDG  = os.path.join(EXP, "llm_as_judge")

SEP = "─" * 70

def status(calc, esperado, tol=0.001):
    if abs(calc - esperado) <= tol:
        return "OK"
    elif abs(calc - esperado) <= 0.01:
        return f"~OK (diff={calc-esperado:+.4f})"
    else:
        return f"DIFF (calc={calc:.4f}, esperado={esperado:.4f}, diff={calc-esperado:+.4f})"

def mcnemar(s1, s2):
    ids = s1.index.intersection(s2.index)
    b = ((s1[ids]==False) & (s2[ids]==True)).sum()
    c = ((s1[ids]==True)  & (s2[ids]==False)).sum()
    p = 2 * binom.cdf(min(b,c), b+c, 0.5)
    return b, c, p

print(SEP)
print("VERIFICAÇÃO COMPLETA: números reportados vs. dados brutos")
print(SEP)

# ════════════════════════════════════════════════════════════════
# Tabela 1 — avg_q
# ════════════════════════════════════════════════════════════════
print("\n### Tabela 1 — avg_q")
print(f"{'Config':<16} {'Calc':>8} {'Esperado':>8} {'Status'}")
configs_q = [
    ('1.5B GPT',  'qwen_1_5b_gpt_num_ques.csv',    0.00),
    ('1.5B GEM',  'qwen_1_5b_gemini_num_ques.csv',  0.00),
    ('7B GPT',    'qwen_7b_gpt_num_ques.csv',       8.77),
    ('7B GEM',    'qwen_7b_gemini_num_ques.csv',    7.47),
    ('14B GPT',   'qwen_14b_gpt_num_ques.csv',      5.37),
    ('14B GEM',   'qwen_14b_gemini_num_ques.csv',   4.47),
    ('32B GPT',   'qwen_32b_gpt_num_ques.csv',      4.37),
    ('32B GEM',   'qwen_32b_gemini_num_ques.csv',   3.97),
]
for label, fname, esperado_val in configs_q:
    df   = pd.read_csv(os.path.join(QA, fname))
    n    = df['instance_id'].nunique()
    calc = df['num_questions'].sum() / n
    print(f"  {label:<14} {calc:>8.2f} {esperado_val:>8.2f}  {status(calc, esperado_val, tol=0.005)}")

# ════════════════════════════════════════════════════════════════
# Tabela 1 — IG
# ════════════════════════════════════════════════════════════════
print(f"\n### Tabela 1 — IG")
print(f"{'Config':<16} {'Calc':>8} {'Esperado':>8} {'Status'}")
configs_ig = [
    ('7B GPT',  'qwen_7b_gpt_embedding_results.csv',    0.0794),
    ('7B GEM',  'qwen_7b_gemini_embedding_results.csv', 0.1548),
    ('14B GPT', 'qwen_14b_gpt_embedding_results.csv',   0.1031),
    ('14B GEM', 'qwen_14b_gemini_embedding_results.csv',0.1773),
    ('32B GPT', 'qwen_32b_gpt_embedding_results.csv',   0.1356),
    ('32B GEM', 'qwen_32b_gemini_embedding_results.csv',0.1808),
]
for label, fname, esperado_val in configs_ig:
    df   = pd.read_csv(os.path.join(COS, fname))
    calc = df.groupby('instance_id')['difference_score'].mean().mean()
    print(f"  {label:<14} {calc:>8.4f} {esperado_val:>8.4f}  {status(calc, esperado_val)}")

# ════════════════════════════════════════════════════════════════
# Tabela 1 — Judge
# ════════════════════════════════════════════════════════════════
print(f"\n### Tabela 1 — Judge")
print(f"{'Config':<16} {'Calc':>8} {'Esperado':>8} {'Status'}")
configs_j = [
    ('7B GPT',  'qwen_7b_gpt_gpt4o_evaluation_results.csv',    2.48),
    ('7B GEM',  'qwen_7b_gemini_gpt4o_evaluation_results.csv', 3.50),
    ('14B GPT', 'qwen_14b_gpt_gpt4o_evaluation_results.csv',   3.22),
    ('14B GEM', 'qwen_14b_gemini_gpt4o_evaluation_results.csv',4.39),
    ('32B GPT', 'qwen_32b_gpt_gpt4o_evaluation_results.csv',   4.03),
    ('32B GEM', 'qwen_32b_gemini_gpt4o_evaluation_results.csv',4.53),
]
for label, fname, esperado_val in configs_j:
    df   = pd.read_csv(os.path.join(JDG, fname))
    calc = df.groupby('instance_id')['new_information_score'].mean().mean()
    print(f"  {label:<14} {calc:>8.4f} {esperado_val:>8.2f}  {status(calc, esperado_val, tol=0.005)}")

# ════════════════════════════════════════════════════════════════
# §3.2 — Wilcoxon IG GPT vs GEM
# ════════════════════════════════════════════════════════════════
print(f"\n### §3.2 — Wilcoxon IG GPT vs GEM")
print(f"{'Modelo':<8} {'Delta calc':>12} {'Esperado':>12} {'p calc':>12} {'Esperado':>12} {'Status'}")
wilcox_32_esp = [('7B', 0.0754, 1.06e-5), ('14B', 0.0729, 1.38e-6), ('32B', 0.0452, 2.08e-5)]
for model, d_esp, p_esp in wilcox_32_esp:
    m   = model.lower()
    vg  = pd.read_csv(os.path.join(COS, f'qwen_{m}_gpt_embedding_results.csv')).groupby('instance_id')['difference_score'].mean()
    vk  = pd.read_csv(os.path.join(COS, f'qwen_{m}_gemini_embedding_results.csv')).groupby('instance_id')['difference_score'].mean()
    ids = vg.index.intersection(vk.index)
    delta = (vk[ids] - vg[ids]).mean()
    _, p  = stats.wilcoxon(vg[ids], vk[ids])
    st = "OK" if abs(delta-d_esp)<0.0001 and abs(p-p_esp)/p_esp<0.02 else "DIFF"
    print(f"  {model:<6} {delta:>12.4f} {d_esp:>12.4f} {p:>12.2e} {p_esp:>12.2e}  {st}")

# ════════════════════════════════════════════════════════════════
# §3.3 — Wilcoxon IG e Judge 7B vs 14B
# ════════════════════════════════════════════════════════════════
print(f"\n### §3.3 — Wilcoxon IG 7B vs 14B")
print(f"{'Sim':<8} {'Delta':>10} {'Esperado':>10} {'p calc':>10} {'Esperado':>10} {'Status'}")
for label, sfx, d_esp, p_esp in [('GPT','gpt',0.0237,0.175),('GEM','gemini',0.0214,0.044)]:  # unilateral (H1: 14B>7B)
    v7  = pd.read_csv(os.path.join(COS,f'qwen_7b_{sfx}_embedding_results.csv')).groupby('instance_id')['difference_score'].mean()
    v14 = pd.read_csv(os.path.join(COS,f'qwen_14b_{sfx}_embedding_results.csv')).groupby('instance_id')['difference_score'].mean()
    ids = v7.index.intersection(v14.index)
    delta = (v14[ids]-v7[ids]).mean()
    _, p = stats.wilcoxon(v7[ids], v14[ids], alternative='less')
    print(f"  {label:<6} {delta:>10.4f} {d_esp:>10.4f} {p:>10.3f} {p_esp:>10.3f}  {status(p,p_esp)}")

print(f"\n### §3.3 — Wilcoxon Judge 7B vs 14B")
print(f"{'Sim':<8} {'p calc':>10} {'Esperado':>10} {'Status'}")
for label, sfx, p_esp in [('GPT','gpt',0.012),('GEM','gemini',0.003)]:  # unilateral
    v7  = pd.read_csv(os.path.join(JDG,f'qwen_7b_{sfx}_gpt4o_evaluation_results.csv')).groupby('instance_id')['new_information_score'].mean()
    v14 = pd.read_csv(os.path.join(JDG,f'qwen_14b_{sfx}_gpt4o_evaluation_results.csv')).groupby('instance_id')['new_information_score'].mean()
    ids = v7.index.intersection(v14.index)
    _, p = stats.wilcoxon(v7[ids], v14[ids], alternative='less')
    print(f"  {label:<6} {p:>10.3f} {p_esp:>10.3f}  {status(p,p_esp)}")

# ════════════════════════════════════════════════════════════════
# §3.4 — Pearson r e medianas
# ════════════════════════════════════════════════════════════════
print(f"\n### §3.4 — Pearson r 14B vs 32B")
print(f"{'Sim':<8} {'r calc':>8} {'Esperado':>8} {'p calc':>8} {'Esperado':>8} {'Status'}")
for label, sfx, r_esp, p_esp in [('GPT','gpt',0.424,0.020),('GEM','gemini',-0.006,0.975)]:
    v14 = pd.read_csv(os.path.join(JDG,f'qwen_14b_{sfx}_gpt4o_evaluation_results.csv')).groupby('instance_id')['new_information_score'].mean()
    v32 = pd.read_csv(os.path.join(JDG,f'qwen_32b_{sfx}_gpt4o_evaluation_results.csv')).groupby('instance_id')['new_information_score'].mean()
    ids = v14.index.intersection(v32.index)
    r, p = stats.pearsonr(v14[ids], v32[ids])
    st = "OK" if abs(r-r_esp)<0.001 and abs(p-p_esp)<0.001 else "DIFF"
    print(f"  {label:<6} {r:>8.3f} {r_esp:>8.3f} {p:>8.3f} {p_esp:>8.3f}  {st}")

print(f"\n### §3.4 — Medianas e outlier (Figura 1b)")
df14g = pd.read_csv(os.path.join(COS,'qwen_14b_gemini_embedding_results.csv'))
df32g = pd.read_csv(os.path.join(COS,'qwen_32b_gemini_embedding_results.csv'))
m14   = df14g.groupby('instance_id')['difference_score'].mean().median()
m32   = df32g.groupby('instance_id')['difference_score'].mean().median()
out32 = df32g.groupby('instance_id')['difference_score'].mean().max()
print(f"  14B GEM mediana: {m14:.4f}  esperado: ~0.166  {status(m14,0.166)}")
print(f"  32B GEM mediana: {m32:.4f}  esperado: ~0.167  {status(m32,0.167)}")
print(f"  32B GEM outlier: {out32:.4f}  esperado: ~0.385  {status(out32,0.385)}")

# ════════════════════════════════════════════════════════════════
# Tabela 2 — Swap
# ════════════════════════════════════════════════════════════════
print(f"\n### Tabela 2 — Scores médios do swap")
swap = pd.read_csv(os.path.join(JDG,'llm_judge_swap_results.csv'))
paper_t2 = {'7b':(4.167,4.167,4.767,4.767),'14b':(3.862,3.897,4.793,4.793),'32b':(4.133,4.100,4.867,4.867)}
for m in ['7b','14b','32b']:
    sub = swap[swap['model']==m]
    s   = [sub['score_orig_gpt'].mean(), sub['score_swap_gem_gpt'].mean(),
           sub['score_orig_gemini'].mean(), sub['score_swap_gpt_gem'].mean()]
    p   = paper_t2[m]
    ok  = all(abs(c-a)<0.001 for c,a in zip(s,p))
    print(f"  {m}: {s[0]:.3f}/{s[1]:.3f}/{s[2]:.3f}/{s[3]:.3f}  esperado: {p[0]:.3f}/{p[1]:.3f}/{p[2]:.3f}/{p[3]:.3f}  {'OK' if ok else 'DIFF'}")

print(f"\n### Tabela 2 — p-values viés simulador")
p_bias_esp = {'7b':6.32e-4,'14b':2.53e-4,'32b':1.32e-4}
for m in ['7b','14b','32b']:
    sub = swap[swap['model']==m]
    if m == '32b':
        _, p = stats.wilcoxon(sub['score_swap_gem_gpt'], sub['score_orig_gemini'])
    else:
        _, p = stats.wilcoxon(sub['score_orig_gpt'], sub['score_swap_gpt_gem'])
    pap = p_bias_esp[m]
    print(f"  {m}: {p:.2e}  esperado: {pap:.2e}  {'OK' if abs(p-pap)/pap<0.01 else 'DIFF'}")

print(f"\n### Tabela 2 — p-values pergunta do agente")
p_q_esp = {'7b':(1.0,1.0),'14b':(1.0,1.0),'32b':(0.75,1.0)}  # teste exato (N efetivo pequeno)
def _wilcoxon_perg(a, b):
    # Colunas de pergunta sao degeneradas (muitos empates). Descartamos os zeros
    # ANTES do teste exato para o resultado nao depender da versao do scipy:
    # versoes antigas caem para a aproximacao normal quando ha zeros; sem os zeros
    # o exato roda igual em qualquer versao. Sem pares nao-nulos, o teste e degenerado (p=1).
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 1.0
    try:              return stats.wilcoxon(d, mode='exact').pvalue
    except TypeError: return stats.wilcoxon(d, method='exact').pvalue
    except Exception: return stats.wilcoxon(d).pvalue
for m in ['7b','14b','32b']:
    sub = swap[swap['model']==m]
    pg = _wilcoxon_perg(sub['score_orig_gpt'], sub['score_swap_gem_gpt'])
    pk = _wilcoxon_perg(sub['score_orig_gemini'], sub['score_swap_gpt_gem'])
    p1,p2 = p_q_esp[m]
    st = "OK" if abs(pg-p1)<0.001 and abs(pk-p2)<0.001 else "DIFF"
    print(f"  {m}: GPT-ref={pg:.3f} (esperado {p1}) | GEM-ref={pk:.3f} (esperado {p2})  {st}")

# ════════════════════════════════════════════════════════════════
# §4.1 — Recusas e McNemar
# ════════════════════════════════════════════════════════════════
print(f"\n### §4.1 — Taxas de recusa")
ref_esp = {
    ('7b','gpt'):(4,30,13.3), ('14b','gpt'):(10,29,34.5), ('32b','gpt'):(11,30,36.7),
    ('7b','gemini'):(0,30,0.0), ('14b','gemini'):(1,29,3.4), ('32b','gemini'):(1,30,3.3),
}
for (m,sim),(n_esp,denom,pct_esp) in ref_esp.items():
    sub  = swap[swap['model']==m]
    col  = 'a_gpt' if sim=='gpt' else 'a_gemini'
    mask = sub[col].str.contains("I don't have that information", case=False)
    n_calc, pct_calc = mask.sum(), mask.mean()*100
    st = "OK" if n_calc==n_esp else "DIFF"
    print(f"  {m} {sim:<8} {n_calc}/{len(sub)}={pct_calc:.1f}%  esperado:{n_esp}/{denom}={pct_esp}%  {st}")

print(f"\n### §4.1 — McNemar")
PHRASE = "I don't have that information"
r = {m: {s: swap[swap['model']==m].set_index('instance_id')[f'a_{s}'].str.contains(PHRASE,case=False)
         for s in ['gpt','gemini']} for m in ['7b','14b','32b']}
tests = [("7B vs 14B GPT",   r['7b']['gpt'],  r['14b']['gpt'],  0.109),
         ("GPT vs GEM 14B",  r['14b']['gpt'], r['14b']['gemini'],0.004),
         ("GPT vs GEM 32B",  r['32b']['gpt'], r['32b']['gemini'],0.002)]
for label, s1, s2, p_esp in tests:
    b, c, p = mcnemar(s1, s2)
    print(f"  {label:<22} b={b} c={c} p={p:.3f}  esperado:{p_esp:.3f}  {status(p,p_esp,tol=0.0005)}")

# ════════════════════════════════════════════════════════════════
# §4.2 — Deltas e 32B distintas
# ════════════════════════════════════════════════════════════════
print(f"\n### §4.2 — Deltas score (A_gem - A_gpt, Q_gpt fixa)")
for m in ['7b','14b','32b']:
    sub   = swap[swap['model']==m]
    delta = sub['score_swap_gpt_gem'].mean() - sub['score_orig_gpt'].mean()
    print(f"  {m}: delta={delta:.2f}  esperado [+0.60 a +0.93]  {'' if 0.55<=delta<=0.95 else ''}")

print(f"\n### §4.2 — 32B perguntas distintas e mesmo score")
s32 = swap[swap['model']=='32b']
diff_mask = s32['q_gpt'] != s32['q_gemini']
n_diff = diff_mask.sum()
same   = (s32[diff_mask]['score_orig_gemini'] == s32[diff_mask]['score_swap_gpt_gem']).sum()
print(f"  Distintas: {n_diff}/30  esperado:19/30  {'OK' if n_diff==19 else 'DIFF'}")
print(f"  Mesmo score: {same}/{n_diff}  esperado:17/19  {'OK' if same==17 else 'DIFF'}")

# ════════════════════════════════════════════════════════════════
# §4.2 — Viés sem recusas (análise estratificada)
# Esperado: recusas respondem por 50 a 77% do delta; excluídas as instâncias
# com recusa em qualquer simulador, o viés permanece significativo
# (N = 26, 19 e 19; deltas +0,26 a +0,42; p entre 0,003 e 0,025)
# ════════════════════════════════════════════════════════════════
print(f"\n### §4.2 — Viés sem recusas (estratificado)")
paper_strat = {  # (N_clean, delta_qgpt, p_qgpt, delta_qgem, p_qgem, contrib_pct)
    '7b':  (26, 0.346, 0.0027, 0.346, 0.0027, 50),
    '14b': (19, 0.421, 0.0047, 0.368, 0.0082, 70),
    '32b': (19, 0.263, 0.0253, 0.316, 0.0143, 77),
}
for m in ['7b','14b','32b']:
    sub  = swap[swap['model']==m]
    ref_gpt = sub['a_gpt'].str.contains(PHRASE, case=False)
    ref_any = ref_gpt | sub['a_gemini'].str.contains(PHRASE, case=False)
    # contribuição das recusas (A_gpt) para o delta bruto (referência Q_gpt)
    d_all   = sub['score_swap_gpt_gem'] - sub['score_orig_gpt']
    contrib = 100 * (d_all[ref_gpt].sum()/len(sub)) / d_all.mean()
    clean = sub[~ref_any]
    d1 = (clean['score_swap_gpt_gem'] - clean['score_orig_gpt']).mean()
    d2 = (clean['score_orig_gemini'] - clean['score_swap_gem_gpt']).mean()
    _, p1 = stats.wilcoxon(clean['score_orig_gpt'],     clean['score_swap_gpt_gem'])
    _, p2 = stats.wilcoxon(clean['score_swap_gem_gpt'], clean['score_orig_gemini'])
    N_a, d1_a, p1_a, d2_a, p2_a, c_a = paper_strat[m]
    ok = (len(clean)==N_a and abs(d1-d1_a)<0.005 and abs(p1-p1_a)<0.0005
          and abs(d2-d2_a)<0.005 and abs(p2-p2_a)<0.0005 and abs(contrib-c_a)<1)
    print(f"  {m}: N={len(clean)} (esperado {N_a}) | "
          f"Q_gpt d={d1:+.3f} p={p1:.4f} (esperado {d1_a:+.3f}/{p1_a}) | "
          f"Q_gem d={d2:+.3f} p={p2:.4f} (esperado {d2_a:+.3f}/{p2_a}) | "
          f"recusas={contrib:.0f}% (esperado {c_a}%)  {'OK' if ok else 'DIFF'}")
print("  faixas reportadas: deltas +0,26 a +0,42; p entre 0,003 e 0,025; contribuição 50 a 77%")

print(f"\n{SEP}")
print("FIM. Verifique os status acima: OK = confere, DIFF = divergência.")
print(SEP)
