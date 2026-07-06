from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = str(Path(__file__).resolve().parents[1])

def generate_original_boxplot(simulator_name, cosine_files, judge_files, output_filename):
    models = ["Qwen 7B", "Qwen 14B", "Qwen 32B"]
    
    # Carrega os dados brutos (cada turno é uma linha)
    difference_dataframes = [pd.read_csv(f) for f in cosine_files]
    annotation_dataframes = [pd.read_csv(f) for f in judge_files]
    
    # Concatena sem agrupar pela média (metodologia Ambig-SWE)
    difference_combined = pd.concat(
        [pd.DataFrame({'Model': models[i], 'Distance': df['difference_score']})
         for i, df in enumerate(difference_dataframes)]
    ).reset_index(drop=True)
    
    annotation_combined = pd.concat(
        [pd.DataFrame({'Model': models[i], 'Score': df['new_information_score']})
         for i, df in enumerate(annotation_dataframes)]
    ).reset_index(drop=True)
    
    plt.rcParams.update({'font.size': 14})
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=False)
    
    palette = sns.color_palette("Set2", len(models))
    
    sns.boxplot(ax=axes[0], x='Model', y='Distance', data=difference_combined, palette=palette)
    axes[0].set_title(f'Information Gain (Cosine Distance) - {simulator_name}')  
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Cosine Distance')
    
    axes[0].text(-0.1, 1.05, r'$\mathbf{(a)}$', transform=axes[0].transAxes, fontsize=16, fontweight='bold')
    
    sns.boxplot(ax=axes[1], x='Model', y='Score', data=annotation_combined, palette=palette, showmeans=False)
    axes[1].set_title(f'LLM-as-a-Judge Score - {simulator_name}')  
    axes[1].set_xlabel('')
    axes[1].set_ylabel('LLM-as-Judge Score')
    
    axes[1].text(-0.1, 1.05, r'$\mathbf{(b)}$', transform=axes[1].transAxes, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_filename, format="pdf")
    plt.close()
    print(f"Gerado: {output_filename}")


if __name__ == "__main__":
    output_dir = f"{base_dir}/figures"
    os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # GPT-mini
    # ==========================================
    gpt_cosine_files = [
        f"{base_dir}/cosine_distance/qwen_7b_gpt_embedding_results.csv",
        f"{base_dir}/cosine_distance/qwen_14b_gpt_embedding_results.csv",
        f"{base_dir}/cosine_distance/qwen_32b_gpt_embedding_results.csv"
    ]
    gpt_judge_files = [
        f"{base_dir}/llm_as_judge/qwen_7b_gpt_gpt4o_evaluation_results.csv",
        f"{base_dir}/llm_as_judge/qwen_14b_gpt_gpt4o_evaluation_results.csv",
        f"{base_dir}/llm_as_judge/qwen_32b_gpt_gpt4o_evaluation_results.csv"
    ]
    generate_original_boxplot(
        "GPT-mini", 
        gpt_cosine_files, 
        gpt_judge_files, 
        f"{output_dir}/qwen_boxplot_ambig_gpt.pdf"
    )

    # ==========================================
    # Gemini Flash
    # ==========================================
    gemini_cosine_files = [
        f"{base_dir}/cosine_distance/qwen_7b_gemini_embedding_results.csv",
        f"{base_dir}/cosine_distance/qwen_14b_gemini_embedding_results.csv",
        f"{base_dir}/cosine_distance/qwen_32b_gemini_embedding_results.csv"
    ]
    gemini_judge_files = [
        f"{base_dir}/llm_as_judge/qwen_7b_gemini_gpt4o_evaluation_results.csv",
        f"{base_dir}/llm_as_judge/qwen_14b_gemini_gpt4o_evaluation_results.csv",
        f"{base_dir}/llm_as_judge/qwen_32b_gemini_gpt4o_evaluation_results.csv"
    ]
    generate_original_boxplot(
        "Gemini Flash", 
        gemini_cosine_files, 
        gemini_judge_files, 
        f"{output_dir}/qwen_boxplot_ambig_gemini.pdf"
    )
