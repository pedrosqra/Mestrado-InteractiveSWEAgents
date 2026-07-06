import os
import pandas as pd
from scipy.stats import wilcoxon

def run_wilcoxon(file1, file2, col_name, expected_p, name):
    if not os.path.exists(file1) or not os.path.exists(file2):
        print(f"FAIL: missing file for {name}")
        return
        
    df1 = pd.read_csv(file1).groupby('instance_id')[col_name].mean().reset_index()
    df2 = pd.read_csv(file2).groupby('instance_id')[col_name].mean().reset_index()
    
    merged = pd.merge(df1, df2, on='instance_id')
    # Unilateral, como H1 do Ambig-SWE original (d~ > 0). file1=7B, file2=14B;
    # 'less' testa 7B < 14B, ou seja, a maior escala produz metrica superior.
    stat, p_val = wilcoxon(merged[f"{col_name}_x"], merged[f"{col_name}_y"], alternative='less')
    
    is_sig = p_val < 0.05
    exp_sig = expected_p < 0.05
    
    if is_sig == exp_sig:
        print(f"PASS: {name} | p={p_val:.3f} (expected ~{expected_p:.3f}) | Significant? {is_sig}")
    else:
        print(f"FAIL: {name} | p={p_val:.3f} (expected ~{expected_p:.3f}) | Significant mismatch")


def main():
    print("=== Script 4: Validate Wilcoxon Scale ===")
    base_dir = ".." if os.path.exists("../experiments") else "."
    
    # Cosine distance
    gpt_7b = os.path.join(base_dir, "experiments/cosine_distance/qwen_7b_gpt_embedding_results.csv")
    gpt_14b = os.path.join(base_dir, "experiments/cosine_distance/qwen_14b_gpt_embedding_results.csv")
    gemini_7b = os.path.join(base_dir, "experiments/cosine_distance/qwen_7b_gemini_embedding_results.csv")
    gemini_14b = os.path.join(base_dir, "experiments/cosine_distance/qwen_14b_gemini_embedding_results.csv")
    
    run_wilcoxon(gpt_7b, gpt_14b, "difference_score", 0.175, "Scale (7B vs 14B) - GPT-mini")
    run_wilcoxon(gemini_7b, gemini_14b, "difference_score", 0.044, "Scale (7B vs 14B) - Gemini Flash")
    
    # LLM Judge
    judge_gpt_7b = os.path.join(base_dir, "experiments/llm_as_judge/qwen_7b_gpt_gpt4o_evaluation_results.csv")
    judge_gpt_14b = os.path.join(base_dir, "experiments/llm_as_judge/qwen_14b_gpt_gpt4o_evaluation_results.csv")
    judge_gemini_7b = os.path.join(base_dir, "experiments/llm_as_judge/qwen_7b_gemini_gpt4o_evaluation_results.csv")
    judge_gemini_14b = os.path.join(base_dir, "experiments/llm_as_judge/qwen_14b_gemini_gpt4o_evaluation_results.csv")

    run_wilcoxon(judge_gpt_7b, judge_gpt_14b, "new_information_score", 0.012, "Judge Scale (7B vs 14B) - GPT-mini")
    run_wilcoxon(judge_gemini_7b, judge_gemini_14b, "new_information_score", 0.003, "Judge Scale (7B vs 14B) - Gemini Flash")

if __name__ == "__main__":
    main()
