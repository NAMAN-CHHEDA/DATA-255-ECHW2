# Metrics

No experiment has been run. This file defines the metrics to compute later. Fill the table only with numbers printed by your own evaluation code.

Primary task: binary in-hospital mortality on PhysioNet Challenge 2012. The positive class is in-hospital death.

Configured seed: 8893 (`src/config.py`). This is the seed the code will use. It is not evidence that a run finished.

## Definitions

These are the intended test-set metrics. Validation uses the same definitions for model selection. Do not choose the checkpoint using the test set.

| Metric | Role | Definition to implement |
| --- | --- | --- |
| AUROC | Primary ranking metric | Area under the ROC curve of predicted P(death). Threshold-free. |
| AUPRC | Primary metric under class imbalance | Area under the precision-recall curve, with death as the positive class. |
| Brier score | Probability calibration check | Mean squared error between predicted probability and the binary label. |
| Cross-entropy | Training and reporting loss | Mean binary cross-entropy with logits on the mortality label. |

Early stopping is configured to monitor validation AUPRC (`EARLY_STOPPING_METRIC` in `src/config.py`), higher is better. Accuracy at a 0.5 threshold is not a primary metric for this task.

Compute each metric from model outputs on data held out for that split. Training-split class weights, if you add them later, must be estimated from the training split only.

## Results

Leave these cells as `not run` until `scripts/run_experiment_*.py` finishes on your machine.

| Experiment | Model | Split | AUROC | AUPRC | Brier | Cross-entropy | Checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | GRU baseline | validation | not run | not run | not run | not run | |
| 1 | GRU baseline | test | not run | not run | not run | not run | |
| 2 | MLP Neural ODE | validation | not run | not run | not run | not run | |
| 2 | MLP Neural ODE | test | not run | not run | not run | not run | |
| 3 | KAN Neural ODE | validation | not run | not run | not run | not run | |
| 3 | KAN Neural ODE | test | not run | not run | not run | not run | |
| 4 | Liquid model | validation | not run | not run | not run | not run | |
| 4 | Liquid model | test | not run | not run | not run | not run | |
