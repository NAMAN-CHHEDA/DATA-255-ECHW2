# Part G Reflection Template

Complete a separate final document at `docs/part_g_reflection.md` after the
GPU experiments finish. Each response must be 100–250 words. Clearly separate:

- **Actual evidence:** values, plots, timings, or artifacts produced by your runs.
- **Interpretation:** your explanation of what that evidence means.

Do not replace missing evidence with assumptions.

## 1. Why is feeding Δt to an LSTM not equivalent to a true continuous-time model?

**Target:** 100–250 words.

**Evidence placeholders**
- Experiment 2 robustness table: `outputs/experiment_2/robustness_results.csv`
  - [INSERT GRU RESULTS AT 100%, 75%, 50%, AND 25% RETENTION]
  - [INSERT MLP-ODE, KAN-ODE, AND LTC COMPARISON VALUES]
- Experiment 2 figure: `outputs/experiment_2/robustness_auroc.png`
  - [DESCRIBE AN ACTUAL TREND VISIBLE IN THE FIGURE]
- [STATE WHICH CLAIMS ARE DIRECT OBSERVATIONS AND WHICH ARE INTERPRETATION]

**My result-based argument**

[WRITE YOUR 100–250 WORD ARGUMENT HERE. Explain why adding elapsed time as an
input does not itself define continuous hidden-state dynamics between events.]

## 2. Why can deeper or wider KANs still help despite the Kolmogorov-Arnold theorem?

**Target:** 100–250 words.

**Evidence placeholders**
- Experiment 4 table: `outputs/experiment_4/parameter_efficiency_results.csv`
  - [INSERT THE TESTED KAN WIDTH/GRID CONFIGURATIONS AND ACTUAL COUNTS]
  - [INSERT THE OBSERVED PERFORMANCE DIFFERENCES ACROSS CAPACITIES]
- Experiment 3 metadata: `outputs/experiment_3/interpretability_metadata.json`
  - [INSERT RELEVANT TRAINED KAN STRUCTURE OR ATTRIBUTION FACTS]
- [DISTINGUISH THE REPRESENTATION THEOREM FROM FINITE-DATA OPTIMIZATION EVIDENCE]

**My result-based argument**

[WRITE YOUR 100–250 WORD ARGUMENT HERE. Do not treat the theorem as an empirical
guarantee about trainability, finite width, noise, optimization, or generalization.]

## 3. How does LTC/CfC forgetfulness compare with the Neural ODE’s continuous dynamics?

**Target:** 100–250 words.

**Evidence placeholders**
- Experiment 1 headline results: `outputs/experiment_1/headline_results.csv`
  - [INSERT LTC AND NEURAL ODE TEST RESULTS]
- Optional CfC result: `outputs/experiment_1/cfc_variant_results.csv`
  - [INSERT CFC RESULT AND TIMING ONLY IF THIS OPTIONAL RUN EXISTS]
- Experiment 2 robustness results: `outputs/experiment_2/robustness_results.csv`
  - [INSERT RETENTION-LEVEL EVIDENCE FOR LTC AND BOTH NEURAL ODES]
- [LABEL MEASURED BEHAVIOR AS EVIDENCE; LABEL MECHANISTIC EXPLANATIONS AS INTERPRETATION]

**My result-based argument**

[WRITE YOUR 100–250 WORD ARGUMENT HERE. Compare adaptive liquid memory/decay
with latent evolution under the learned Neural ODE vector field.]

## 4. Based on Experiment 4, where did KAN-ODE outperform or underperform MLP-ODE?

**Target:** 100–250 words.

**Evidence placeholders**
- Experiment 4 table: `outputs/experiment_4/parameter_efficiency_results.csv`
  - [INSERT EACH TARGET HIDDEN DIMENSION]
  - [INSERT ACTUAL PARAMETER COUNTS AND NONZERO MATCHING DIFFERENCES]
  - [INSERT TEST AUROC/AUPRC/Brier COMPARISONS]
- Experiment 4 figure:
  `outputs/experiment_4/parameter_efficiency_auroc_vs_params.png`
  - [DESCRIBE THE ACTUAL CURVES/MARKERS]
- [DO NOT CLAIM EXACT PARAMETER MATCHING UNLESS THE SAVED DIFFERENCE IS ZERO]

**My result-based argument**

[WRITE YOUR 100–250 WORD ARGUMENT HERE. State separately where KAN-ODE was
better, worse, or inconclusive based on completed runs.]

## 5. Which model should be deployed for bedside early warning?

**Target:** 100–250 words.

**Evidence placeholders**
- Accuracy/calibration: `outputs/experiment_1/headline_results.csv`
  - [INSERT AUROC, AUPRC, AND BRIER EVIDENCE]
- Missingness robustness: `outputs/experiment_2/robustness_results.csv`
  - [INSERT RETENTION-LEVEL EVIDENCE]
- Interpretability: `outputs/experiment_3/interpretability_metadata.json`
  and `outputs/experiment_3/kan_edge_functions/`
  - [INSERT ONLY ACTUAL LATENT/TIME INTERPRETABILITY EVIDENCE]
- Latency/resource evidence: Experiment 1 timing and peak-memory columns
  - [INSERT MEAN EPOCH TIME, PARAMETER COUNT, AND MEMORY EVIDENCE]
- [SEPARATE DEPLOYMENT JUDGMENT FROM OBSERVED METRICS]

**My result-based argument**

[WRITE YOUR 100–250 WORD DEPLOYMENT RECOMMENDATION HERE. Address accuracy,
calibration, interpretability limits, latency, memory, monitoring, and safety.]

## 6. Did the biologically or mathematically principled motivation of LTC or KAN pay off empirically?

**Target:** 100–250 words.

**Evidence placeholders**
- Experiment 1: `outputs/experiment_1/headline_results.csv`
  - [INSERT HEADLINE PERFORMANCE, PARAMETERS, TIMING, AND MEMORY]
- Experiment 2: `outputs/experiment_2/robustness_results.csv`
  - [INSERT IRREGULAR-SAMPLING ROBUSTNESS EVIDENCE]
- Experiment 3: `outputs/experiment_3/interpretability_metadata.json`
  - [INSERT REAL PYKAN ATTRIBUTION/SYMBOLIC FACTS; DO NOT INVENT CLINICAL MAPPINGS]
- Experiment 4: `outputs/experiment_4/parameter_efficiency_results.csv`
  - [INSERT PARAMETER-EFFICIENCY EVIDENCE]
- [STATE WHETHER THE EVIDENCE SUPPORTS, PARTLY SUPPORTS, OR DOES NOT SUPPORT THE MOTIVATION]

**My result-based argument**

[WRITE YOUR 100–250 WORD ARGUMENT HERE. Keep empirical findings distinct from
theoretical or biological motivation.]
