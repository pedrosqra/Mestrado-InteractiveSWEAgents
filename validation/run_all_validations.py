import subprocess
import os
import sys

def main():
    scripts = [
        "validate_raw_data.py",
        "validate_aggregation.py",
        "validate_wilcoxon_simulator.py",
        "validate_wilcoxon_scale.py",
        "validate_mcnemar_1_5b.py",
        "validate_swap_experiment.py"
    ]
    
    total_passes = 0
    total_fails = 0
    
    with open("validation_report.txt", "w") as f:
        f.write("=== Validation Report ===\n\n")
        
        for script in scripts:
            if not os.path.exists(script):
                msg = f"ERROR: Script {script} not found.\n"
                print(msg)
                f.write(msg)
                continue
                
            print(f"Running {script}...")
            f.write(f"--- Running {script} ---\n")
            
            result = subprocess.run([sys.executable, script], capture_output=True, text=True)
            output = result.stdout
            if result.stderr:
                output += "\n[STDERR]\n" + result.stderr
            
            f.write(output)
            f.write("\n")
            
            passes = output.count("PASS:")
            fails = output.count("FAIL:")
            
            total_passes += passes
            total_fails += fails
            
        summary = f"\n=== Summary ===\nPASS: {total_passes}\nFAIL: {total_fails}\n"
        print(summary)
        f.write(summary)

if __name__ == "__main__":
    main()
