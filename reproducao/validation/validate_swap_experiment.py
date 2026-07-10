import os
import pandas as pd
from scipy.stats import wilcoxon

def main():
    print("=== Script 6: Validate Swap Experiment ===")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "..", "..")
    
    swap_file = os.path.join(base_dir, "reproducao/experiments/llm_as_judge/llm_judge_swap_results.csv")
    if not os.path.exists(swap_file):
        print(f"FAIL: {swap_file} not found")
        return
        
    df = pd.read_csv(swap_file)
    
    models = ["7b", "14b", "32b"]
    
    expected_simulator = {
        "7b": 6.32e-4,
        "14b": 2.53e-4,
        "32b": 1.32e-4
    }
    
    expected_question = {
        "7b": 1.000,
        "14b": 0.317,
        "32b": 0.564
    }
    
    for model in models:
        df_model = df[df['model'] == model].copy()
        if df_model.empty:
            print(f"FAIL: no data for {model}")
            continue
            
        q_identical = (df_model['q_gpt'] == df_model['q_gemini']).sum()
        total = len(df_model)
        print(f"\n[{model}] Identical questions: {q_identical}/{total}")
        
        if model in ["qwen_7b", "qwen_14b"]:
            if q_identical == total:
                print(f"PASS: {model} questions are 100% identical")
            else:
                print(f"FAIL: {model} questions are not 100% identical")
        elif model == "qwen_32b":
            diff = total - q_identical
            if diff == 19:
                print(f"PASS: {model} questions differed in exactly 19 cases")
            else:
                print(f"FAIL: {model} differed in {diff} cases, expected 19")
        
        # Effect of simulator (fix Q, change A): compare Original_GPT with Swap_Gemini_GPT
        # In Swap_Gem_GPT, Q is from GPT, A is from Gemini. So Q is fixed, A changed to Gemini.
        try:
            stat_sim, p_sim = wilcoxon(df_model['score_orig_gpt'], df_model['score_swap_gpt_gem'], alternative='two-sided')
        except ValueError:
            p_sim = 1.0
            
        exp_sim = expected_simulator[model]
        if (p_sim < 0.05) == (exp_sim < 0.05):
            print(f"PASS: Simulator effect {model} p={p_sim:.2e} (expected ~{exp_sim:.2e})")
        else:
            print(f"FAIL: Simulator effect {model} p={p_sim:.2e} (expected ~{exp_sim:.2e})")
            
        # Effect of question (change Q, fix A): compare Original_Gemini with Swap_Gemini_GPT
        # Swap_Gem_GPT = Q_gpt, A_gem. Original_Gemini = Q_gem, A_gem. So A is fixed (Gemini).
        try:
            stat_q, p_q = wilcoxon(df_model['score_orig_gemini'], df_model['score_swap_gpt_gem'], alternative='two-sided')
        except ValueError:
            p_q = 1.0
            
        exp_q = expected_question[model]
        if (p_q < 0.05) == (exp_q < 0.05) or abs(p_q - exp_q) < 0.05:
            print(f"PASS: Question effect {model} p={p_q:.3f} (expected ~{exp_q:.3f})")
        else:
            print(f"FAIL: Question effect {model} p={p_q:.3f} (expected ~{exp_q:.3f})")

if __name__ == "__main__":
    main()
