import os
import pandas as pd
from scipy.stats import wilcoxon

def main():
    print("=== Script 3: Validate Wilcoxon Simulator ===")
    base_dir = ".." if os.path.exists("../experiments") else "."
    
    expected_p = {
        "7b": 1.06e-5,
        "14b": 1.38e-6,
        "32b": 2.07e-5
    }

    sizes = ["7b", "14b", "32b"]
    for size in sizes:
        gpt_file = os.path.join(base_dir, f"experiments/cosine_distance/qwen_{size}_gpt_embedding_results.csv")
        gemini_file = os.path.join(base_dir, f"experiments/cosine_distance/qwen_{size}_gemini_embedding_results.csv")
        
        df_gpt = pd.read_csv(gpt_file).groupby('instance_id')['difference_score'].mean().reset_index()
        df_gemini = pd.read_csv(gemini_file).groupby('instance_id')['difference_score'].mean().reset_index()
        
        merged = pd.merge(df_gpt, df_gemini, on='instance_id', suffixes=('_gpt', '_gemini'))
        
        expected_n = 29 if size == "14b" else 30
        if len(merged) == expected_n:
            print(f"PASS: {size} has N={len(merged)} valid paired issues")
        else:
            print(f"FAIL: {size} has N={len(merged)} paired issues, expected {expected_n}")
            
        stat, p_val = wilcoxon(merged['difference_score_gpt'], merged['difference_score_gemini'], alternative='two-sided')
        
        expected = expected_p[size]
        if p_val < 0.0001:
            print(f"PASS: {size} p-value {p_val:.2e} < 0.0001 (expected ~{expected:.2e})")
        else:
            print(f"FAIL: {size} p-value {p_val:.2e} >= 0.0001")

if __name__ == "__main__":
    main()
