"""
Reproduz a análise de validade de construto do IG (sensibilidade do IG ao comprimento da resposta):
verbosidade vs. Ganho de Informação.

Definições (importante):
- Por instância: IG = MEDIA do difference_score; comprimento = SOMA do tamanho
  (em caracteres) das respostas do simulador naquela instância.
- Correlação comprimento~IG: Spearman (não paramétrico).
- Controle por comprimento: regressão linear OLS de IG sobre [comprimento, is_gemini]
  por escala, e agregada com dummies de escala. Esta regressão é a unica analise
  paramétrica do estudo, usada como analise de sensibilidade (ver Secao 2.2 e 5.4).

Uso: python3 verbosity_ig_analysis.py
Requer: pandas, numpy, scipy
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
COS = os.path.join(BASE, "..", "cosine_distance")


def per_instance(model, sim):
    d = pd.read_csv(os.path.join(COS, f"qwen_{model}_{sim}_embedding_results.csv"))
    d["alen"] = d["answer"].astype(str).str.len()
    g = d.groupby("instance_id").agg(ig=("difference_score", "mean"),
                                     length=("alen", "sum"))
    return d, g


def ols_coef_p(X, y, idx):
    """OLS manual; retorna (coef, p bilateral) da coluna idx."""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    sigma2 = (resid @ resid) / (n - k)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    t = beta[idx] / se[idx]
    p = 2 * stats.t.sf(abs(t), n - k)
    return beta[idx], p


def main():
    print("=== Secao 5.4: verbosidade vs IG ===\n")

    print("Comprimento medio da resposta por turno (caracteres) e IG medio:")
    for m in ["7b", "14b", "32b"]:
        for sim in ["gpt", "gemini"]:
            d, g = per_instance(m, sim)
            print(f"  {m:>3} {sim:<7} len/turno={d['alen'].mean():>6.0f}  IG={g['ig'].mean():.4f}")

    print("\nSpearman (comprimento total por instancia ~ IG):")
    for m in ["7b", "14b", "32b"]:
        for sim in ["gpt", "gemini"]:
            _, g = per_instance(m, sim)
            rho, p = stats.spearmanr(g["length"], g["ig"])
            print(f"  {m:>3} {sim:<7} rho={rho:+.3f}  p={p:.3f}")

    print("\nRegressao IG ~ comprimento + is_gemini (efeito do simulador, ajustado):")
    pooled = []
    for m in ["7b", "14b", "32b"]:
        _, gg = per_instance(m, "gpt"); gg["is_gem"] = 0
        _, gk = per_instance(m, "gemini"); gk["is_gem"] = 1
        G = pd.concat([gg, gk], ignore_index=True)
        G["scale"] = m
        pooled.append(G)
        X = np.column_stack([np.ones(len(G)), G["length"].values, G["is_gem"].values])
        coef, p = ols_coef_p(X, G["ig"].values, 2)
        print(f"  {m:>3}: efeito Gemini ajustado = {coef:+.4f}  (p={p:.3f}, N={len(G)})")

    P = pd.concat(pooled, ignore_index=True)
    X = np.column_stack([np.ones(len(P)), P["length"].values, P["is_gem"].values,
                         (P["scale"] == "14b").astype(int), (P["scale"] == "32b").astype(int)])
    coef, p = ols_coef_p(X, P["ig"].values, 2)
    print(f"  agregado (com dummies de escala) = {coef:+.4f}  (p={p:.2e})")


if __name__ == "__main__":
    main()
