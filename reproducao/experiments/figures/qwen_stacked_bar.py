from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

base_dir = str(Path(__file__).resolve().parents[1])

# Define os modelos e seus respectivos arquivos CSV
datasets = [
    {"label": "Claude 3.5 Sonnet\n(Ambig-SWE Baseline)", "file": f"{base_dir}/llm_as_judge/claude_sonnet_gpt4o_evaluation_results_strict.csv"},
    {"label": "Qwen 7B (GPT-mini)", "file": f"{base_dir}/llm_as_judge/qwen_7b_gpt_gpt4o_evaluation_results.csv"},
    {"label": "Qwen 14B (GPT-mini)", "file": f"{base_dir}/llm_as_judge/qwen_14b_gpt_gpt4o_evaluation_results.csv"},
    {"label": "Qwen 32B (GPT-mini)", "file": f"{base_dir}/llm_as_judge/qwen_32b_gpt_gpt4o_evaluation_results.csv"},
    {"label": "Qwen 7B (Gemini)", "file": f"{base_dir}/llm_as_judge/qwen_7b_gemini_gpt4o_evaluation_results.csv"},
    {"label": "Qwen 14B (Gemini)", "file": f"{base_dir}/llm_as_judge/qwen_14b_gemini_gpt4o_evaluation_results.csv"},
    {"label": "Qwen 32B (Gemini)", "file": f"{base_dir}/llm_as_judge/qwen_32b_gemini_gpt4o_evaluation_results.csv"}
]

data = []
for item in datasets:
    if os.path.exists(item["file"]):
        df = pd.read_csv(item["file"])
        # Calcula a porcentagem de cada nota (1 a 5)
        counts = df['new_information_score'].value_counts(normalize=True) * 100
        
        # Garante que as 5 notas existam no dict, mesmo que sejam 0%
        counts_dict = {score: counts.get(score, 0) for score in [1, 2, 3, 4, 5]}
        counts_dict["Model"] = item["label"]
        data.append(counts_dict)

# Cria o DataFrame para o gráfico
df_plot = pd.DataFrame(data).set_index("Model")
# Inverte a ordem das linhas para o Claude ficar no topo do gráfico horizontal
df_plot = df_plot.iloc[::-1]

# Paleta de cores: Vermelho escuro (1) até Azul escuro (5)
# Isso ajuda visualmente a separar falhas de sucessos
colors = ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4'] 

fig, ax = plt.subplots(figsize=(12, 7))

# Plota o gráfico de barras empilhadas 100% (horizontal)
df_plot.plot(kind='barh', stacked=True, color=colors, ax=ax, edgecolor='white', width=0.7)

ax.set_title("Distribution of Turn-Level Information Gain Scores", fontsize=16, pad=20)
ax.set_xlabel("Percentage of Turns (%)", fontsize=14)
ax.set_ylabel("")
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='x', labelsize=12)
ax.set_xlim(0, 100)

# Adiciona o texto de porcentagem dentro das barrinhas
for c in ax.containers:
    # Só exibe o número se a barra for maior que 4% para não amontoar texto
    labels = [f'{v.get_width():.1f}%' if v.get_width() > 4 else '' for v in c]
    ax.bar_label(c, labels=labels, label_type='center', fontsize=10, color='black', fontweight='bold')

# Arruma a legenda para ficar fora do gráfico
ax.legend(title='Judge Score\n(1=Refusal, 5=Max Info)', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11, title_fontsize=12)

plt.tight_layout()
output_path = f"{base_dir}/figures/qwen_stacked_bar_chart.pdf"
plt.savefig(output_path, format='pdf', bbox_inches='tight')
print(f"Gerado: {output_path}")
