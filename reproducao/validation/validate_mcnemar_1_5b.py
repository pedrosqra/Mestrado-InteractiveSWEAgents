import os
import pandas as pd
from scipy.stats import binom

def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return 2 * binom.cdf(k, n, 0.5)

def main():
    print("=== Script 5: Validate McNemar 1.5B ===")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "..", "..")
    
    sample_file = os.path.join(base_dir, "data/sample_30_underspecified.csv")
    if not os.path.exists(sample_file):
        print("FAIL: sample_30_underspecified.csv not found")
        return
        
    all_issues = pd.read_csv(sample_file)['instance_id'].tolist()
    
    # Read 1.5B GPT and 7B GPT
    f_1_5b = os.path.join(base_dir, "reproducao/experiments/cosine_distance/qwen_1_5b_gpt_embedding_results.csv")
    f_7b = os.path.join(base_dir, "reproducao/experiments/cosine_distance/qwen_7b_gpt_embedding_results.csv")
    
    try:
        df_1_5b = pd.read_csv(f_1_5b)
        if 'question' in df_1_5b.columns:
            has_1_5b = set(df_1_5b.dropna(subset=['question'])['instance_id'].unique())
        else:
            has_1_5b = set(df_1_5b['instance_id'].unique())
    except pd.errors.EmptyDataError:
        has_1_5b = set()
        
    df_7b = pd.read_csv(f_7b)
    if 'question' in df_7b.columns:
        has_7b = set(df_7b.dropna(subset=['question'])['instance_id'].unique())
    else:
        has_7b = set(df_7b['instance_id'].unique())
    
    interact_1_5b = [1 if i in has_1_5b else 0 for i in all_issues]
    interact_7b = [1 if i in has_7b else 0 for i in all_issues]
    
    a = sum(1 for i, j in zip(interact_1_5b, interact_7b) if i == 1 and j == 1)
    b = sum(1 for i, j in zip(interact_1_5b, interact_7b) if i == 1 and j == 0)
    c = sum(1 for i, j in zip(interact_1_5b, interact_7b) if i == 0 and j == 1)
    d = sum(1 for i, j in zip(interact_1_5b, interact_7b) if i == 0 and j == 0)
    
    print("Contingency Table:")
    print(f"           7B: Yes   7B: No")
    print(f"1.5B: Yes    {a}         {b}")
    print(f"1.5B: No     {c}         {d}")
    
    pvalue = mcnemar_exact(b, c)
    
    if pvalue < 0.0001:
        print(f"PASS: McNemar p-value {pvalue:.2e} < 0.0001")
    else:
        print(f"FAIL: McNemar p-value {pvalue:.2e} >= 0.0001")

if __name__ == "__main__":
    main()
