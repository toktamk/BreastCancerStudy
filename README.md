
# Multimodal Survival Modeling for Breast Cancer (METABRIC)

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Status](https://img.shields.io/badge/status-research%20pipeline-success)
![Dataset](https://img.shields.io/badge/dataset-METABRIC-informational)
![Task](https://img.shields.io/badge/task-survival%20%2B%205yr%20risk-orange)
![Model](https://img.shields.io/badge/model-penalized%20CoxPH-purple)

<!-- Key results (recalibrated CoxPH @ 60 months; n=380 defined labels) -->
![AUROC](https://img.shields.io/badge/AUROC-0.967-brightgreen)
![AUROC%2095%25%20CI](https://img.shields.io/badge/AUROC%2095%25%20CI-0.949%E2%80%930.981-brightgreen)
![Brier](https://img.shields.io/badge/Brier-0.064-brightgreen)
![Brier%2095%25%20CI](https://img.shields.io/badge/Brier%2095%25%20CI-0.047%E2%80%930.082-brightgreen)

![Calibration](https://img.shields.io/badge/evaluation-calibration%20%7C%20fairness%20%7C%20robustness-blueviolet)
![Bootstrap](https://img.shields.io/badge/robustness-bootstrap%201000x-blue)
![Fairness](https://img.shields.io/badge/fairness-subgroup%20parity%20checks-blue)

A reproducible, calibration-aware, fairness-evaluated framework for 5-year overall survival (OS) risk prediction and censoring-aware survival modeling using multimodal clinical + omics data from METABRIC.



## What this project is

This repository provides an end-to-end experimental framework for:

- **Survival modeling** (penalized Cox PH)
- **Fixed-horizon (5-year) risk prediction** at **60 months**
- **Calibration diagnostics** (slope/intercept, ECE, reliability bins)
- **Fairness evaluation** across clinically relevant subgroups
- **Robustness assessment** via **bootstrap confidence intervals**
- **Multimodal ablation** and **missing-modality stress tests**
- **Governed train/val/test splits** (patient-level, deterministic seeds)



## Key results (current best run)

Recalibrated CoxPH survival model evaluated on the test split (defined labels **n=380**):

- **AUROC:** 0.967  
- **AUROC 95% CI (bootstrap, 1000×):** 0.949–0.981  
- **Brier:** 0.064  
- **Brier 95% CI (bootstrap, 1000×):** 0.047–0.082  
- **Mean predicted risk vs prevalence:** 0.188 vs 0.216 (mild underprediction)

Subgroup fairness diagnostics show consistently high discrimination across age strata, ER status, molecular subtype, and menopausal status, with moderate and biologically plausible calibration heterogeneity in smaller molecular subtypes.



## Repository structure

```
data/
  raw/
  processed/
    cohorts/
      metabric_complete_case_v1.parquet
      metabric_complete_case_v1.manifest.json

src/
  data/
  splits/
  models/
    survival.py
    baselines.py
  eval/
    evaluate.py
    survival_eval.py
    time_dependent.py
  fairness/
    group_eval.py
  robustness/
    bootstrap.py
  experiments/
    ablation.py

scripts/
  etl_build_processed.py
  build_cohort.py
  run_survival_models.py
  run_baselines.py
  sanity_check_calibration.py
  eval_survival_fairness_robustness.py

runs/
  binary/
  survival/

tests/
```



## End-to-end workflow

### 1) ETL for generating processed tables

```bash
python scripts/etl_build_processed.py \
  --raw_dir data/raw \
  --out_dir data/processed
```

### 2) Build cohort (complete-case multimodal)

```bash
python scripts/build_cohort.py \
  --processed_dir data/processed \
  --out_path data/processed/cohorts/metabric_complete_case_v1.parquet \
  --require_expr \
  --require_cna
```

### 3) Train and evaluate survival model

```bash
python scripts/run_survival_models.py \
  --cohort data/processed/cohorts/metabric_complete_case_v1.parquet \
  --outdir runs/survival/coxph
```

### 4) Binary baseline (fixed-horizon y60)

```bash
python scripts/run_baselines.py \
  --cohort data/processed/cohorts/metabric_complete_case_v1.parquet \
  --label y60 \
  --outdir runs/binary/lr
```



## Calibration evaluation (sanity checks)

Survival horizon calibration + invariants:

```bash
python scripts/sanity_check_calibration.py \
  --cohort data/processed/cohorts/metabric_complete_case_v1.parquet \
  --preds runs/survival/coxph/preds/test.parquet \
  --p-col risk_prob_horizon \
  --outdir runs/survival/coxph
```

Outputs include Brier, log loss, AUROC, calibration slope/intercept, ECE, reliability bins, and label/horizon consistency checks.



## Fairness and robustness evaluation (subgroup and bootstrap)

```bash
python scripts/eval_survival_fairness_robustness.py \
  --cohort data/processed/cohorts/metabric_complete_case_v1.parquet \
  --preds runs/survival/coxph/preds/test.parquet \
  --p-col risk_prob_horizon \
  --group-cols age_group,SEX,ER_IHC,CLAUDIN_SUBTYPE,INFERRED_MENOPAUSAL_STATE \
  --threshold 0.2 \
  --bootstrap 1000 \
  --outdir runs/survival/coxph/fairness_robustness
```

Produces:

- `report.json` (global + subgroup metrics)
- `fairness_<COL>_group_metrics.csv` (AUROC/AUPRC/Brier + calibration per group)
- `fairness_<COL>_threshold_parity.csv` (TPR/FPR/PPV parity at threshold)

> Note: `age_group` is derived from `AGE_AT_DIAGNOSIS` if not present.



## Recalibration workflow

If calibration slope deviates from 1:

1. Fit logistic recalibration on validation predictions  
2. Apply transformation to test predictions  
3. Save recalibrated probabilities under `risk_prob_horizon_recal`  
4. Re-run fairness + robustness evaluation using `--p-col risk_prob_horizon_recal`

Example:

```bash
python scripts/eval_survival_fairness_robustness.py \
  --cohort data/processed/cohorts/metabric_complete_case_v1.parquet \
  --preds runs/survival/coxph/preds/test_recal.parquet \
  --p-col risk_prob_horizon_recal \
  --group-cols age_group,SEX,ER_IHC,CLAUDIN_SUBTYPE,INFERRED_MENOPAUSAL_STATE \
  --threshold 0.2 \
  --bootstrap 1000 \
  --outdir runs/survival/coxph/fairness_robustness_recal
```



## Governance and reproducibility

- Patient-level splits (no overlap across train/val/test)
- Deterministic seeds and stored split CSVs
- Train-only preprocessing fit
- Explicit failure on invalid stratification and broken joins
- Cohort manifest JSON for traceability and regeneration



## Testing

```bash
pytest -q
```



## Intended use

Designed for:

- Academic survival modeling research
- Clinical ML benchmarking
- Fairness-aware prognostic modeling
- Calibration methodology and diagnostics

Not intended for direct clinical deployment.
