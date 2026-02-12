# BreastCancerStudy

## Dataset: BRCA_METABRIC (Molecular Taxonomy of Breast Cancer International Consortium)

This project uses the METABRIC breast cancer cohort obtained from cBioPortal (Study ID: brca_metabric). The dataset provides matched clinical and multi-omic tumor profiles with long-term survival follow-up, making it an appropriate benchmark for multimodal cancer risk prediction.

METABRIC contains approximately 2,000 breast cancer patients with:

Structured clinical variables

Gene expression profiles (microarray-based mRNA expression)

Copy number alteration (CNA) data

Overall survival time and event status

This benchmark explicitly treats METABRIC as a multimodal dataset, integrating clinical and molecular representations of tumor biology.
### Download dataset
for downloading the dataset, you can go to the website[https://www.cbioportal.org/study/summary?id=brca_metabric]
then, download Breast Cancer (METABRIC, Nature 2012 & Nat Commun 2016) [https://datahub.assets.cbioportal.org/brca_metabric.tar.gz].

## Benchmark Task
### Primary Task: 5-Year Overall Survival Risk Prediction

The primary objective is to predict the risk of death within 5 years (60 months) of diagnosis.

The task is implemented under two complementary modeling paradigms:

Fixed-horizon binary prediction

Classification of death within 60 months.

Full survival analysis

Time-to-event modeling accounting for right censoring.

This dual formulation enables rigorous evaluation of:

Discrimination (C-index, time-dependent AUC)

Calibration

Fairness across clinical subgroups

Robustness under modality perturbation

## Modalities

The benchmark integrates three data modalities:

Clinical variables (structured tabular data)

Transcriptomic profiles (gene expression)

Genomic structural variation (copy number alterations)

Unimodal, bimodal, and trimodal models are systematically evaluated.

## Data Documentation

Detailed dataset specification, preprocessing rules, label construction procedures, and schema definitions are provided in:

DATA_DESCRIPTION.md

Raw data is treated as immutable and stored in data/raw/. All preprocessing steps are deterministic and version-controlled to ensure full reproducibility.

