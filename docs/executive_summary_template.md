# Executive Summary Template — One Page Maximum

Complete a separate final document at `docs/executive_summary.md` only after
the GPU experiments finish. Do not insert numbers, rankings, or claims until
they are supported by saved experiment artifacts.

## Objective

[STATE THE MORTALITY-PREDICTION OBJECTIVE AND THE MODEL COMPARISON IN 1–2
SENTENCES.]

## Dataset and Protocol

[SUMMARIZE PHYSIONET CHALLENGE 2012 SET A, THE SAVED 70/15/15 PATIENT SPLIT,
THE 12 VARIABLES, TRAIN-ONLY NORMALIZATION, SEED 8893, AND VALIDATION-AUPRC
EARLY STOPPING.]

## Headline Results

[REFERENCE `outputs/experiment_1/headline_results.csv` AND INSERT A CONCISE
RESULT TABLE OR SENTENCE ONLY AFTER THE REAL RUN.]

## Robustness Finding

[REFERENCE `outputs/experiment_2/robustness_results.csv` AND
`robustness_auroc.png`; SUMMARIZE ONE EVIDENCE-BACKED ZERO-SHOT FINDING.]

## Parameter-Efficiency Finding

[REFERENCE `outputs/experiment_4/parameter_efficiency_results.csv` AND
`parameter_efficiency_auroc_vs_params.png`; REPORT ACTUAL COUNTS AND ANY
NONZERO MATCHING DIFFERENCES.]

## Interpretability Finding

[REFERENCE `outputs/experiment_3/interpretability_metadata.json`,
`top_two_input_dimensions.csv`, AND THE KAN EDGE FIGURE. USE LATENT/TIME
LABELS, NOT UNSUPPORTED RAW-CLINICAL ATTRIBUTION.]

## Deployment Recommendation

[RECOMMEND A MODEL ONLY AFTER CONSIDERING SAVED ACCURACY, CALIBRATION,
ROBUSTNESS, INTERPRETABILITY, LATENCY, PARAMETER, AND MEMORY EVIDENCE.]

## Limitations

[LIST THE MOST IMPORTANT DATA, SPLIT, MODELING, COMPUTE, INTERPRETABILITY, AND
EXTERNAL-VALIDATION LIMITATIONS.]
