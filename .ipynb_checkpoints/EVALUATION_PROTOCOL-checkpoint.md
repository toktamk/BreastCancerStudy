# Evaluation Protocol
## 1. Overview

This document defines the evaluation framework for the BRCA_METABRIC multimodal survival benchmark. The protocol is designed to ensure:

Statistical rigor

Clinical relevance

Fairness auditing

Robustness under perturbation

Reproducibility

The benchmark evaluates both:

Fixed-horizon 5-year overall survival (binary classification)

Full time-to-event survival modeling

All reported metrics must include confidence intervals and be computed on held-out test data.

## 2. Data Splitting Strategy
### 2.1 Train / Validation / Test Split

Stratified by event indicator

Fixed random seed

No patient overlap across splits

Recommended split:

60% training

20% validation

20% test

### 2.2 Cross-Validation

For model comparison:

5-fold cross-validation on training set

Final model retrained on train+validation

Final evaluation on held-out test set

All splits are deterministic and version-controlled.

## 3. Discrimination Metrics
### 3.1 Binary 5-Year Risk Prediction

Primary metrics:

AUROC

AUPRC

Balanced Accuracy

F1-score at clinically selected threshold

Secondary metrics:

Sensitivity and specificity at fixed risk threshold

Net reclassification improvement (optional)

95% confidence intervals:

Bootstrapped (≥1000 resamples)

### 3.2 Full Survival Modeling

Primary metric:

Concordance Index (C-index)

Secondary metrics:

Time-dependent AUC

Integrated Brier Score

Confidence intervals:

Bootstrap with patient-level resampling

## 4. Calibration Assessment

Calibration is mandatory for clinical risk modeling.

### 4.1 Binary 5-Year Risk

Metrics:

Calibration curve (reliability diagram)

Expected Calibration Error (ECE)

Brier score

Calibration slope

Calibration intercept

Models may optionally undergo:

Temperature scaling

Isotonic regression

Calibration must be reported:

Before recalibration

After recalibration

### 4.2 Survival Models

Metrics:

Calibration at 5 years

Integrated calibration index (if applicable)

Brier score over time

Risk stratification plots must show:

Predicted risk deciles

Kaplan–Meier curves per risk group

## 5. Fairness Evaluation

Fairness is evaluated as performance consistency across clinically meaningful subgroups.

### 5.1 Subgroup Axes

Primary subgroup analyses:

Age groups (e.g., ≤50 vs >50)

Tumor stage

ER status

HER2 status

Subgroups must have sufficient sample size for reliable estimation.

### 5.2 Fairness Metrics

For each subgroup:

AUROC

C-index

Brier score

Calibration slope

Sensitivity at fixed threshold

Fairness gap defined as:

Δ_metric = max(metric_group_i) − min(metric_group_i)


We report:

Discrimination disparity

Calibration disparity

Error rate disparity

### 5.3 Threshold-Based Fairness

At a clinically selected risk threshold:

Compare false positive rates

Compare false negative rates

Evaluate decision-level disparity

No post-hoc threshold adjustment across groups is performed unless explicitly stated.

## 6. Robustness Evaluation

Robustness tests simulate real-world deployment instability.

### 6.1 Modality Ablation

Evaluate performance when:

Removing gene expression

Removing CNA

Using clinical-only

This quantifies modality contribution.

### 6.2 Modality Missing Simulation

Simulate missing modalities during inference:

Random gene dropout

CNA masking

Partial feature corruption

Measure degradation in:

AUROC

C-index

Calibration

### 6.3 Noise Injection

Add controlled Gaussian noise to gene expression features to assess stability.

Report performance degradation curve.

### 6.4 Distribution Shift Simulation

If applicable:

Train on subset (e.g., early enrollment)

Test on different subset (e.g., later enrollment)

Measure shift impact on discrimination and calibration.

## 7. Uncertainty Quantification

Models must report uncertainty estimates.

Approaches may include:

Deep ensembles

Monte Carlo dropout

Bootstrap aggregation

Metrics:

Prediction variance

Risk confidence intervals

Coverage of prediction intervals (if using conformal methods)

## 8. Statistical Comparison of Models

Model comparisons must include:

Paired bootstrap tests

DeLong test for AUROC

Confidence interval overlap analysis

A model is considered superior only if improvement is statistically significant.

## 9. Reporting Standards

All results must include:

Point estimate

95% confidence interval

Number of patients evaluated

Event rate in test set

Tables must distinguish:

Internal validation performance

Final held-out test performance

## 10. Reproducibility Requirements

Each experiment must log:

Random seed

Data split version

Feature selection strategy

Hyperparameters

Software versions

All figures and tables must be reproducible from:

scripts/evaluate.py


No manual result editing is permitted.

## 11. Minimum Benchmark Report

Each model must provide:

Discrimination table

Calibration plot

Subgroup fairness table

Robustness degradation plot

Risk stratification Kaplan–Meier curves

## 12. Success Criteria for Multimodal Integration

A multimodal model is considered beneficial if it demonstrates:

Statistically significant improvement in C-index or AUROC

Equal or improved calibration

No increase in subgroup disparity

Acceptable robustness under perturbation

Performance improvement without calibration or fairness preservation is not considered sufficient.

## Summary

This evaluation protocol ensures that the BRCA_METABRIC benchmark is:

Statistically rigorous

Clinically meaningful

Fairness-aware

Robustness-aware

Fully reproducible

It is designed to move beyond conventional machine learning benchmarking and reflect standards expected in translational clinical AI research.