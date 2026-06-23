"""Lance tout le pipeline dans l'ordre. Usage : python run_all.py"""
import runpy
import sys
from pathlib import Path

STEPS = [
    "01_load_data.py",
    "02_strategies.py",
    "03_metrics.py",
    "04_factor_attribution.py",
    "05_benchmark_dependence.py",
    "06_funds.py",
    "07_tests_regression.py",
    "08_concentration_subperiods.py",
    "09_mag7_budget_frontier.py",
]

here = Path(__file__).resolve().parent
for step in STEPS:
    print(f"\n{'='*70}\n>>> {step}\n{'='*70}")
    runpy.run_path(str(here / step), run_name="__main__")
print("\nPipeline termine.")
