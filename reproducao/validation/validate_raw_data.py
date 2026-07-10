import os
import pandas as pd

def check_file_exists(filepath):
    if not os.path.exists(filepath):
        print(f"FAIL: File not found: {filepath}")
        return False
    return True

def main():
    print("=== Script 1: Validate Raw Data ===")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "..", "..")

    sample_file = os.path.join(base_dir, "data/sample_30_underspecified.csv")
    if not check_file_exists(sample_file):
        return

    sample_df = pd.read_csv(sample_file)
    n_unique = sample_df['instance_id'].nunique()
    
    if n_unique == 30:
        print("PASS: sample_30_underspecified.csv has exactly 30 unique instance_ids")
    else:
        print(f"FAIL: sample_30_underspecified.csv has {n_unique} unique instance_ids, expected 30")

    batteries = [
        "qwen_1_5b_gpt", "qwen_1_5b_gemini", 
        "qwen_7b_gpt", "qwen_7b_gemini", 
        "qwen_14b_gpt", "qwen_14b_gemini", 
        "qwen_32b_gpt", "qwen_32b_gemini"
    ]

    for battery in batteries:
        csv_file = os.path.join(base_dir, f"reproducao/experiments/cosine_distance/{battery}_embedding_results.csv")
        if not check_file_exists(csv_file):
            continue
            
        try:
            df = pd.read_csv(csv_file)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=['instance_id', 'difference_score'])

        if 'question' in df.columns:
            valid_df = df.dropna(subset=['question'])
        else:
            valid_df = df
            
        if 'difference_score' in valid_df.columns:
            has_zeros = (valid_df['difference_score'] == 0.0).any()
            if has_zeros:
                print(f"FAIL: {battery} has difference_score = 0.0")
            else:
                print(f"PASS: {battery} has no difference_score = 0.0")
                
        qa_pairs = valid_df.groupby('instance_id').size() if not valid_df.empty and 'instance_id' in valid_df.columns else pd.Series(dtype=int)
        
        if "1_5b" in battery:
            total_qa = len(valid_df)
            if total_qa == 0:
                print(f"PASS: {battery} has 0 qa_pairs in all 30 issues")
            else:
                print(f"FAIL: {battery} should have 0 qa_pairs, but found {total_qa}")
        elif battery == "qwen_14b_gemini":
            if len(qa_pairs) == 29:
                print(f"PASS: {battery} has exactly 29 valid issues (1 without qa_pairs)")
            else:
                print(f"FAIL: {battery} has {len(qa_pairs)} valid issues, expected 29")

if __name__ == "__main__":
    main()
