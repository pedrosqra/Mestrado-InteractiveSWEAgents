from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = str(Path(__file__).resolve().parents[1])

# Ordem alternada e apenas modelos Qwen
datasets = [
    {"label": "Qwen 7B\n(GPT-mini)", "cosine": f"{base_dir}/cosine_distance/qwen_7b_gpt_embedding_results.csv", "judge": f"{base_dir}/llm_as_judge/qwen_7b_gpt_gpt4o_evaluation_results.csv"},
    {"label": "Qwen 7B\n(Gemini)", "cosine": f"{base_dir}/cosine_distance/qwen_7b_gemini_embedding_results.csv", "judge": f"{base_dir}/llm_as_judge/qwen_7b_gemini_gpt4o_evaluation_results.csv"},
    {"label": "Qwen 14B\n(GPT-mini)", "cosine": f"{base_dir}/cosine_distance/qwen_14b_gpt_embedding_results.csv", "judge": f"{base_dir}/llm_as_judge/qwen_14b_gpt_gpt4o_evaluation_results.csv"},
    {"label": "Qwen 14B\n(Gemini)", "cosine": f"{base_dir}/cosine_distance/qwen_14b_gemini_embedding_results.csv", "judge": f"{base_dir}/llm_as_judge/qwen_14b_gemini_gpt4o_evaluation_results.csv"},
    {"label": "Qwen 32B\n(GPT-mini)", "cosine": f"{base_dir}/cosine_distance/qwen_32b_gpt_embedding_results.csv", "judge": f"{base_dir}/llm_as_judge/qwen_32b_gpt_gpt4o_evaluation_results.csv"},
    {"label": "Qwen 32B\n(Gemini)", "cosine": f"{base_dir}/cosine_distance/qwen_32b_gemini_embedding_results.csv", "judge": f"{base_dir}/llm_as_judge/qwen_32b_gemini_gpt4o_evaluation_results.csv"}
]

# 1. Carrega dados do Cosine Distance
cosine_data = []
for item in datasets:
    if os.path.exists(item["cosine"]):
        df = pd.read_csv(item["cosine"])
        for val in df['difference_score']:
            cosine_data.append({"Model": item["label"], "Distance": val})
df_cos = pd.DataFrame(cosine_data)

# 2. Carrega dados do Judge
judge_data = []
for item in datasets:
    if os.path.exists(item["judge"]):
        df = pd.read_csv(item["judge"])
        counts = df['new_information_score'].value_counts(normalize=True) * 100
        counts_dict = {score: counts.get(score, 0) for score in [1, 2, 3, 4, 5]}
        counts_dict["Model"] = item["label"]
        judge_data.append(counts_dict)

df_judge = pd.DataFrame(judge_data).set_index("Model")
# Inverte a ordem apenas para o gráfico horizontal renderizar de cima para baixo
df_judge = df_judge.iloc[::-1]

# ==================================
# PLOTAGEM DA FIGURA COMPOSTA
# ==================================
fig, axes = plt.subplots(2, 1, figsize=(14, 12))
plt.subplots_adjust(hspace=0.4)

# Paleta alternando tons claros (GPT-mini) e escuros (Gemini) para cada tamanho
box_palette = {
    "Qwen 7B\n(GPT-mini)": "#ffb07c", "Qwen 7B\n(Gemini)": "#ff7f0e", 
    "Qwen 14B\n(GPT-mini)": "#98df8a", "Qwen 14B\n(Gemini)": "#2ca02c", 
    "Qwen 32B\n(GPT-mini)": "#aec7e8", "Qwen 32B\n(Gemini)": "#1f77b4"
}

# (a) Boxplot de Cosine Distance
sns.boxplot(ax=axes[0], x='Model', y='Distance', data=df_cos, palette=box_palette, showfliers=False)
axes[0].set_title("Information Gain (Cosine Distance)", fontsize=16, pad=15)
axes[0].set_xlabel("")
axes[0].set_ylabel("Cosine Distance", fontsize=14)
axes[0].tick_params(axis='x', labelsize=12)
axes[0].tick_params(axis='y', labelsize=12)
axes[0].text(-0.08, 1.05, r'$\mathbf{(a)}$', transform=axes[0].transAxes, fontsize=18, fontweight='bold')
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# (b) Barras empilhadas 100% para o Judge
colors = ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4'] 
df_judge.plot(kind='barh', stacked=True, color=colors, ax=axes[1], edgecolor='white', width=0.7)
axes[1].set_title("Distribution of Turn-Level Information Gain Scores (LLM-as-a-Judge)", fontsize=16, pad=15)
axes[1].set_xlabel("Percentage of Turns (%)", fontsize=14)
axes[1].set_ylabel("")
axes[1].tick_params(axis='y', labelsize=12)
axes[1].tick_params(axis='x', labelsize=12)
axes[1].set_xlim(0, 100)

for c in axes[1].containers:
    labels = [f'{v.get_width():.1f}%' if v.get_width() > 4 else '' for v in c]
    axes[1].bar_label(c, labels=labels, label_type='center', fontsize=10, color='black', fontweight='bold')

axes[1].legend(title='Judge Score (1=Refusal, 5=Max Info)', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11, title_fontsize=12)
axes[1].text(-0.08, 1.05, r'$\mathbf{(b)}$', transform=axes[1].transAxes, fontsize=18, fontweight='bold')

output_path = f"{base_dir}/figures/qwen_composite_figure.pdf"
plt.savefig(output_path, format='pdf', bbox_inches='tight')
print(f"Gerado: {output_path}")
