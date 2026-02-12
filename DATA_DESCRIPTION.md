# Data Description and Specification
## 1. Dataset Source

Dataset: METABRIC (brca_metabric)
Source: cBioPortal Datasets Export
Study ID: brca_metabric

The dataset is downloaded from the official cBioPortal datasets page and stored in:

data/raw/


Raw files are treated as immutable. All derived datasets are generated programmatically.

## 2. Cohort Overview

The METABRIC cohort contains approximately 2,000 breast cancer patients with:

Baseline clinical variables

Gene expression profiles (microarray-based)

Copy number alteration (CNA) data

Long-term overall survival follow-up

Each molecular sample is linked to a unique patient identifier.

## 3. Expected Raw Files

The pipeline expects the following files within data/raw/:

Clinical Files

data_clinical_patient.txt

data_clinical_sample.txt

meta_clinical_patient.txt

meta_clinical_sample.txt

Gene Expression

data_mrna.txt

meta_mrna.txt

Copy Number Alterations

data_cna.txt

meta_cna.txt

File naming may vary slightly depending on portal export version; semantic equivalence is required.

## 4. Modalities
### 4.1 Clinical Modality

Includes structured patient- and tumor-level variables such as:

Age at diagnosis

Tumor size

Stage

Grade

ER / PR / HER2 status

Lymph node status

Treatment indicators (if available)

Overall survival time

Overall survival status

These variables serve as:

Baseline predictors

Fairness subgroup axes

Calibration evaluation strata

### 4.2 Transcriptomic Modality

Gene expression matrix:

Rows: genes

Columns: tumor samples

Values: normalized or z-scored expression levels

High-dimensional molecular representation of tumor biology.

### 4.3 Copy Number Alteration Modality

Gene-level CNA matrix:

Rows: genes

Columns: tumor samples

Values: discrete or continuous CNA measurements

Represents structural genomic alterations complementary to expression data.

## 5. Outcome Definition
### 5.1 Survival Variables

Let:

T = observed survival time in months

E = event indicator (1 = death, 0 = censored)

These are derived directly from clinical patient data.

### 5.2 Primary Endpoint: 5-Year Overall Survival

Prediction horizon: 60 months

Binary label definition:

y60 = 1 if E = 1 and T ≤ 60

y60 = 0 if T > 60

If E = 0 and T < 60, the case is right-censored before horizon.

Handling strategy:

Primary benchmark: exclude censored-before-horizon cases.

Sensitivity analysis: incorporate using inverse probability of censoring weighting (IPCW).

### 5.3 Secondary Endpoint

Full time-to-event survival modeling using (T, E) without horizon restriction.

## 6. Inclusion Criteria

Available survival time and status

Available clinical variables

Available gene expression data

Available CNA data

## 7. Data Processing Principles

Raw files are never modified.

Patient-level merging is deterministic.

Feature engineering is version-controlled.

All derived labels are programmatically generated.

Random seeds are fixed for reproducibility.

Data splits are stratified and documented.

## 8. Reproducibility and Integrity

SHA256 checksums of raw files are stored in data/raw/checksums.txt.

Processed data is stored in data/processed/.

All preprocessing steps are reproducible via scripted pipelines.

No manual data manipulation is permitted.