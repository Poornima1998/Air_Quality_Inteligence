# Experiment records

`experiment_plan.csv` is a plan only. Every row is marked `PLANNED_NOT_RUN_REQUIRES_UCI_DATA`; it is not evidence that the experiment was executed.

After the real UCI CSV is downloaded, `python scripts/run_pipeline.py` writes:

- `experiment_log.csv` – consolidated ANN and baseline records
- `ann_experiment_log.csv` – detailed ANN run outcomes
- `baseline_search.csv` – classical baseline configurations and cross-validation scores
- `ann_repeated_seeds.csv` – repeated selected-ANN runs
- per-configuration learning curves, optimizer comparison and hyperparameter-effect tables

The run-time log is authoritative. Keep failed runs; do not delete them or insert unmeasured scores. Synthetic tests verify code execution only and are not included as project experiments.
