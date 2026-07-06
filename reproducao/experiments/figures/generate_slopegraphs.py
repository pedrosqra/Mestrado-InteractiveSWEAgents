"""
Gera a Figura 2 dos resultados (qwen_slopegraph_cosine e qwen_slopegraph_judge).

Trajetórias por issue (IG e LLM-as-a-Judge), agregadas POR INSTÂNCIA
(média por instance_id), a mesma unidade da Tabela 1 e dos testes estatísticos.
Linhas sólidas azuis: transição controlada de escala local (7B -> 14B).
Linhas tracejadas laranjas: transição de sistema (14B -> 32B via API).
As linhas grossas de média usam apenas instâncias com valores nas três
escalas (N=30 sob GPT-mini; N=29 sob Gemini Flash, pela issue
django__django-13112 ausente no 14B).

Origem: consolidado de scratch/generate_visualizations.py (paths absolutos
para fora do repositório) para este diretório, com paths relativos.

Uso: python3 generate_slopegraphs.py
(o script deve estar em reproducao/experiments/figures/)
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

out_dir = os.path.dirname(os.path.abspath(__file__))          # .../experiments/figures
base_dir = os.path.dirname(out_dir)                            # .../experiments
cosine_dir = f"{base_dir}/cosine_distance"
judge_dir = f"{base_dir}/llm_as_judge"

CONFIGS = [('7B', '7b'), ('14B', '14b'), ('32B', '32b')]


def load_data(directory, suffix, column):
    data = []
    for model, m in CONFIGS:
        for sim, s in [('GPT-mini', 'gpt'), ('Gemini Flash', 'gemini')]:
            df = pd.read_csv(f"{directory}/qwen_{m}_{s}_{suffix}.csv")
            agg = df.groupby('instance_id')[column].mean().reset_index()
            for _, r in agg.iterrows():
                data.append({'model': model, 'simulator': sim,
                             'instance_id': r['instance_id'], 'value': r[column]})
    return pd.DataFrame(data)


df_cos = load_data(cosine_dir, 'embedding_results', 'difference_score')
df_judge = load_data(judge_dir, 'gpt4o_evaluation_results', 'new_information_score')

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 14
})


def generate_slopegraph(df, title, ylabel, ylim=None):
    pivoted = df.pivot(index=['instance_id', 'simulator'], columns='model', values='value').reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    simulators = ['GPT-mini', 'Gemini Flash']
    for idx, sim in enumerate(simulators):
        ax = axes[idx]
        sim_df = pivoted[pivoted['simulator'] == sim].dropna(subset=['7B', '14B', '32B'])

        for _, row in sim_df.iterrows():
            y = [row['7B'], row['14B'], row['32B']]
            x = [0, 1, 2]

            # Segmento 1 (7B -> 14B): comparacao de escala (solida)
            ax.plot(x[:2], y[:2], color='#2b7bba', alpha=0.35, linewidth=1.5, linestyle='-')
            # Segmento 2 (14B -> 32B): comparacao de sistema (tracejada)
            ax.plot(x[1:], y[1:], color='#e38454', alpha=0.35, linewidth=1.5, linestyle='--')

            ax.scatter(x, y, color=['#2b7bba', '#4fa8ad', '#e38454'], s=20, zorder=5)

        # Linhas grossas de media (subconjunto pareado nas tres escalas)
        mean_7b = sim_df['7B'].mean()
        mean_14b = sim_df['14B'].mean()
        mean_32b = sim_df['32B'].mean()

        ax.plot([0, 1], [mean_7b, mean_14b], color='blue', alpha=0.9, linewidth=3.5, linestyle='-', label='Média Escala (Local)')
        ax.plot([1, 2], [mean_14b, mean_32b], color='red', alpha=0.9, linewidth=3.5, linestyle='--', label='Média Sistema (API)')

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['7B\n(Local)', '14B\n(Local)', '32B\n(API)'])
        ax.set_title(f"{title} - {sim}")
        if idx == 0:
            ax.set_ylabel(ylabel)
            ax.legend(loc='lower left')
        if ylim:
            ax.set_ylim(ylim)

    plt.tight_layout()
    return fig


fig_slope_cos = generate_slopegraph(df_cos, "Information Gain ($IG$)", "Semantic Distance (Cosine)")
fig_slope_cos.savefig(f"{out_dir}/qwen_slopegraph_cosine.png", dpi=300)
fig_slope_cos.savefig(f"{out_dir}/qwen_slopegraph_cosine.pdf", format='pdf')
plt.close(fig_slope_cos)

fig_slope_jd = generate_slopegraph(df_judge, "LLM-as-a-Judge Score", "Score (1 to 5)", ylim=(0.8, 5.2))
fig_slope_jd.savefig(f"{out_dir}/qwen_slopegraph_judge.png", dpi=300)
fig_slope_jd.savefig(f"{out_dir}/qwen_slopegraph_judge.pdf", format='pdf')
plt.close(fig_slope_jd)

print(f"Slopegraphs regenerados em {out_dir}/qwen_slopegraph_[cosine,judge].[pdf,png]")
