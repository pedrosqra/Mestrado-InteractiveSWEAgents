# Validation Scripts

This directory contains independent Python scripts to validate the data pipeline and the reported statistical tests.

## Prerequisites

- Python 3.x
- `pandas`
- `numpy`
- `scipy`
- `statsmodels`

You can install them via pip if needed:
```bash
pip install pandas numpy scipy statsmodels
```

## Running the scripts

You can run all validations in sequence using the runner script:
```bash
python run_all_validations.py
```
This will create a `validation_report.txt` file containing the output of all checks, along with a final summary of `PASS/FAIL` counts.

Alternatively, each script is independent and can be run individually:

- `python validate_raw_data.py`: Checks initial assertions on raw CSVs.
- `python validate_aggregation.py`: Checks the Information Gain calculation.
- `python validate_wilcoxon_simulator.py`: Checks Wilcoxon test for simulator effect.
- `python validate_wilcoxon_scale.py`: Checks Wilcoxon test for model scale effect.
- `python validate_mcnemar_1_5b.py`: Checks the McNemar test for 1.5B model interaction.
- `python validate_swap_experiment.py`: Checks the LLM Judge Swap Experiment.
