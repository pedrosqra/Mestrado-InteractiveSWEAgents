from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = str(Path(__file__).resolve().parents[1])
out_dir = f"{base_dir}/figures"
os.makedirs(out_dir, exist_ok=True)

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
            for val in df_c['difference_score']:
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

# ========================================================
# FIGURA 1: BOXPLOT (Cosine Distance) - Lado a Lado
# ========================================================
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

for i, sim in enumerate(["GPT-mini", "Gemini Flash"]):
    ax = axes1[i]
    subset = df_cos[df_cos["Simulator"] == sim]
    
    sns.boxplot(ax=ax, x="Model", y="Distance", data=subset, palette=palette, showfliers=False)
    ax.set_title(f"Information Gain (Cosine Distance) - {sim}", fontsize=13, pad=15)
    ax.set_xlabel("")
    if i == 0:
        ax.set_ylabel("Cosine Distance", fontsize=12)
    else:
        ax.set_ylabel("")
        
    ax.tick_params(axis='x', labelsize=11)
    
    # Linha tracejada separando 7B/14B do 32B
    ax.axvline(1.5, color='gray', linestyle='--', linewidth=1.5)
    
    # Textos de anotação
    ax.text(0.25, -0.15, "Local MLX (4-bit)", transform=ax.transAxes, ha='center', color='#2b7bba', fontsize=11, fontweight='bold')
    ax.text(0.75, -0.15, "OpenRouter\n(precisão não informada)", transform=ax.transAxes, ha='center', color='#e38454', fontsize=11, fontweight='bold')
    
    ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
fig1.subplots_adjust(bottom=0.22)
out_cos = f"{out_dir}/qwen_ig_boxplots_separated.pdf"
fig1.savefig(out_cos, format='pdf')
plt.close(fig1)
print(f"Gerado: {out_cos}")

# ========================================================
# FIGURA 2: STACKED BAR (Judge Score) - Lado a Lado
# ========================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

for i, sim in enumerate(["GPT-mini", "Gemini Flash"]):
    ax = axes2[i]
    subset = df_judge[df_judge["Simulator"] == sim].set_index("Model")
    
    # Para o pandas barh, a última linha do dataframe fica no topo. 
    # Queremos 7B no topo, 14B no meio, 32B na base. 
    # Logo, a ordem no dataframe deve ser 32B, 14B, 7B.
    subset = subset.reindex(["Qwen 32B", "Qwen 14B", "Qwen 7B"])
    
    subset[[1,2,3,4,5]].plot(kind='barh', stacked=True, color=colors_bar, ax=ax, edgecolor='white', width=0.6)
    
    ax.set_title(f"LLM-as-a-Judge Score - {sim}", fontsize=13, pad=15)
    ax.set_xlabel("Percentage of Turns (%)", fontsize=11)
    if i == 0:
        ax.set_ylabel("")
    else:
        ax.set_ylabel("")
        
    ax.set_xlim(0, 100)
    ax.tick_params(axis='y', labelsize=12)
    
    # Linha tracejada horizontal separando 14B do 32B (y=0.5 divide os índices 0 e 1 do barh)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5)
    
    # Adicionar porcentagens nas barras
    for c in ax.containers:
        labels = [f'{v.get_width():.1f}%' if v.get_width() > 4 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center', fontsize=9, color='black', fontweight='bold')
    
    # Textos de anotação na direita
    # O eixo y das barras vai de 0 (32B) até 2 (7B).
    if i == 1:
        # Colocamos o texto só no gráfico da direita para não poluir
        ax.text(1.03, 0.75, "Local MLX\n(4-bit)", transform=ax.transAxes, va='center', color='#2b7bba', fontsize=10, fontweight='bold')
        ax.text(1.03, 0.15, "OpenRouter\n(precisão não\ninformada)", transform=ax.transAxes, va='center', color='#e38454', fontsize=10, fontweight='bold')
        
        # Legenda fora do gráfico
        ax.legend(title='Judge Score\n(1=Refusal, 5=Max Info)', bbox_to_anchor=(1.35, 1), loc='upper left', fontsize=10)
    else:
        ax.get_legend().remove()

plt.tight_layout()
out_judge = f"{out_dir}/qwen_judge_stackedbars_separated.pdf"
fig2.savefig(out_judge, format='pdf')
plt.close(fig2)
print(f"Gerado: {out_judge}")
