"""
Gera a Figura 1 dos resultados (qwen_separated_boxplot):
  - Topo (a/b): boxplots do Ganho de Informação (IG) POR INSTÂNCIA
    (média do difference_score por instance_id), a mesma unidade usada
    na Tabela 1 e em todos os testes estatísticos, e a mesma
    unidade das Figuras 5/6 do Ambig-SWE original. Outliers visíveis
    (showfliers=True), como no artigo original.
  - Base (c/d): stacked bars com a distribuição das notas do LLM-as-a-Judge
    por turno (par pergunta-resposta), em % de turnos.

Uso: python3 generate_separated_boxplot.py
(paths relativos: o script deve estar em reproducao/experiments/figures/)
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

out_dir = os.path.dirname(os.path.abspath(__file__))          # .../experiments/figures
base_dir = os.path.dirname(out_dir)                            # .../experiments

def get_data():
    datasets = [
        {"sim": "GPT-mini", "model": "Qwen 7B", "cos": f"{base_dir}/cosine_distance/qwen_7b_gpt_embedding_results.csv", "judge": f"{base_dir}/llm_as_judge/qwen_7b_gpt_gpt4o_evaluation_results.csv"},
        {"sim": "GPT-mini", "model": "Qwen 14B", "cos": f"{base_dir}/cosine_distance/qwen_14b_gpt_embedding_results.csv", "judge": f"{base_dir}/llm_as_judge/qwen_14b_gpt_gpt4o_evaluation_results.csv"},
        {"sim": "GPT-mini", "model": "Qwen 32B", "cos": f"{base_dir}/cosine_distance/qwen_32b_gpt_embedding_results.csv", "judge": f"{base_dir}/llm_as_judge/qwen_32b_gpt_gpt4o_evaluation_results.csv"},
        {"sim": "Gemini Flash", "model": "Qwen 7B", "cos": f"{base_dir}/cosine_distance/qwen_7b_gemini_embedding_results.csv", "judge": f"{base_dir}/llm_as_judge/qwen_7b_gemini_gpt4o_evaluation_results.csv"},
        {"sim": "Gemini Flash", "model": "Qwen 14B", "cos": f"{base_dir}/cosine_distance/qwen_14b_gemini_embedding_results.csv", "judge": f"{base_dir}/llm_as_judge/qwen_14b_gemini_gpt4o_evaluation_results.csv"},
        {"sim": "Gemini Flash", "model": "Qwen 32B", "cos": f"{base_dir}/cosine_distance/qwen_32b_gemini_embedding_results.csv", "judge": f"{base_dir}/llm_as_judge/qwen_32b_gemini_gpt4o_evaluation_results.csv"},
    ]

    cos_data = []
    judge_data = []
    for d in datasets:
        if os.path.exists(d["cos"]):
            df_c = pd.read_csv(d["cos"])
            # IG por instância: média do deslocamento por par dentro da issue
            # (mesma agregação da Tabela 1 e dos testes de Wilcoxon)
            inst = df_c.groupby('instance_id')['difference_score'].mean()
            for val in inst:
                cos_data.append({"Simulator": d["sim"], "Model": d["model"], "Distance": val})
        if os.path.exists(d["judge"]):
            df_j = pd.read_csv(d["judge"])
            counts = df_j['new_information_score'].value_counts(normalize=True) * 100
            j_dict = {score: counts.get(score, 0) for score in [1, 2, 3, 4, 5]}
            j_dict["Simulator"] = d["sim"]
            j_dict["Model"] = d["model"]
            judge_data.append(j_dict)

    return pd.DataFrame(cos_data), pd.DataFrame(judge_data)

df_cos, df_judge = get_data()

# Paletas de Cores
palette = {"Qwen 7B": "#2b7bba", "Qwen 14B": "#4fa8ad", "Qwen 32B": "#e38454"}
colors_bar = ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4']

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
plt.subplots_adjust(hspace=0.4, wspace=0.25)

# ========================================================
# LINHA SUPERIOR: Boxplots (IG por instância)
# ========================================================
for i, sim in enumerate(["GPT-mini", "Gemini Flash"]):
    ax = axes[0, i]
    subset = df_cos[df_cos["Simulator"] == sim]

    sns.boxplot(ax=ax, x="Model", y="Distance", data=subset, palette=palette,
                showfliers=True, width=0.4,
                flierprops=dict(marker='o', markerfacecolor='none',
                                markeredgecolor='gray', markersize=5))
    ax.set_title(f"({chr(97 + i)}) Information Gain (Cosine Distance) - {sim}", fontsize=13, pad=15)
    ax.set_xlabel("")

    if i == 0:
        ax.set_ylabel("Cosine Distance", fontsize=12)
    else:
        ax.set_ylabel("")

    ax.tick_params(axis='x', labelsize=11)
    ax.set_ylim(0, 0.42)

    # Linha tracejada separando 7B/14B do 32B
    ax.axvline(1.5, color='gray', linestyle='--', linewidth=1.5)

    ax.text(0.25, -0.16, "Local MLX (4-bit)", transform=ax.transAxes, ha='center', color='#2b7bba', fontsize=10, fontweight='bold')
    ax.text(0.75, -0.16, "OpenRouter\n(precisão não informada)", transform=ax.transAxes, ha='center', color='#e38454', fontsize=10, fontweight='bold')

    ax.grid(axis='y', linestyle='--', alpha=0.5)

# ========================================================
# LINHA INFERIOR: Stacked Bars (Judge Score por turno)
# ========================================================
for i, sim in enumerate(["GPT-mini", "Gemini Flash"]):
    ax = axes[1, i]
    subset = df_judge[df_judge["Simulator"] == sim].set_index("Model")

    subset = subset.reindex(["Qwen 32B", "Qwen 14B", "Qwen 7B"])

    subset[[1, 2, 3, 4, 5]].plot(kind='barh', stacked=True, color=colors_bar, ax=ax, edgecolor='white', width=0.5)

    ax.set_title(f"({chr(99 + i)}) LLM-as-a-Judge Score - {sim}", fontsize=13, pad=15)
    ax.set_xlabel("Percentage of Turns (%)", fontsize=11)
    ax.set_ylabel("")
    ax.set_xlim(0, 100)
    ax.tick_params(axis='y', labelsize=11)

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5)

    for c in ax.containers:
        labels = [f'{v.get_width():.1f}%' if v.get_width() > 4 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center', fontsize=9, color='black', fontweight='bold')

    ax.grid(axis='x', linestyle='--', alpha=0.5)

    if i == 1:
        ax.text(1.03, 0.75, "Local MLX\n(4-bit)", transform=ax.transAxes, va='center', color='#2b7bba', fontsize=10, fontweight='bold')
        ax.text(1.03, 0.25, "OpenRouter\n(precisão não\ninformada)", transform=ax.transAxes, va='center', color='#e38454', fontsize=10, fontweight='bold')
        ax.legend(title='Judge Score\n(1=Refusal, 5=Max Info)', bbox_to_anchor=(1.32, 1), loc='upper left', fontsize=10)
    else:
        ax.get_legend().remove()

plt.tight_layout()

plt.savefig(f"{out_dir}/qwen_separated_boxplot.pdf", format='pdf', bbox_inches='tight')
plt.savefig(f"{out_dir}/qwen_separated_boxplot.png", format='png', dpi=300, bbox_inches='tight')

plt.close()
print(f"Figura regenerada em {out_dir}/qwen_separated_boxplot.[pdf,png]")
