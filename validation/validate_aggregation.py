import os
import pandas as pd
import numpy as np

def main():
    print("=== Script 2: Validate Aggregation ===")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "..")
    
    expected_ig = {
        "qwen_7b_gpt": 0.0794,
        "qwen_7b_gemini": 0.1548,
        "qwen_14b_gpt": 0.1031,
        "qwen_14b_gemini": 0.1773,
        "qwen_32b_gpt": 0.1356,
        "qwen_32b_gemini": 0.1808
    }

    for battery, expected in expected_ig.items():
        csv_file = os.path.join(base_dir, f"reproducao/experiments/cosine_distance/{battery}_embedding_results.csv")
        if not os.path.exists(csv_file):
            print(f"FAIL: File not found: {csv_file}")
            continue
            
        df = pd.read_csv(csv_file)
        
        # Valid aggregation (groupby mean)
        issue_means = df.groupby('instance_id')['difference_score'].mean()
        calc_ig = issue_means.mean()
        
        # Buggy aggregation (dict zip)
        buggy_dict = dict(zip(df['instance_id'], df['difference_score']))
        buggy_ig = np.mean(list(buggy_dict.values()))

        diff = abs(calc_ig - expected)
        if diff <= 0.001:
            print(f"PASS: {battery} expected {expected}, got {calc_ig:.4f}")
        else:
            print(f"FAIL: {battery} expected {expected}, got {calc_ig:.4f} (diff > 0.001)")

        if abs(calc_ig - buggy_ig) > 1e-6:
            print(f"      CONFIRMED: dict(zip(...)) would produce {buggy_ig:.4f}, which is different from groupby().mean()")
        else:
            print(f"      INFO: dict(zip(...)) produced same as groupby().mean() for this data.")

if __name__ == "__main__":
    main()
