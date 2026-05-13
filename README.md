# Abstract–Conclusion Relevance Detection (ACRD)

A biomedical text-pair classification pipeline built on PMC Open Access full-text articles.

## Overview

This project detects whether an **abstract** and a **conclusion** belong to the **same biomedical subdomain**. The pipeline includes:

* PMC Open Access data extraction from XML
* abstract and conclusion parsing
* MeSH/tag collection and subdomain grouping
* pair-level dataset generation
* training of multiple deep learning models
* model comparison and evaluation
* packaged inference for later reuse

The benchmarked models include:

* SBERT + MLP
* BioBERT
* Longformer
* BigBird
* T5
* BART

---

## Project Structure

```text
ACRD-Project/
├── notebooks/
│   ├── 01_extract_pmc.ipynb
│   ├── 02_build_dataset.ipynb
│   ├── 03_train_models.ipynb
│   └── 04_inference_demo.ipynb
├── saved_models/
│   ├── SBERT/
│   ├── BioBERT/
│   ├── Longformer/
│   ├── BigBird/
│   ├── T5/
│   └── BART/
└── README.md
```

---

## What the Project Does

The goal is to support biomedical document understanding by learning a binary decision:

* **1 = Same subdomain**
* **0 = Different subdomain**

The task is built from the abstract and conclusion sections of full-text biomedical papers. This is useful for:

* literature screening
* search relevance filtering
* topic-based grouping
* scientific document organization
* downstream retrieval pipelines

---

## Figure Placement Guide for GitHub README

Below are the recommended image placeholders and where to place them.

### 1. Pipeline Overview

**File:** `figures/pipeline_overview.png`

**Place near:** the top of the README, immediately after the overview.

**Suggested content:** a flowchart showing:
PMC OAI-PMH → XML parsing → abstract/conclusion extraction → MeSH collection → dataset building → model training → evaluation → packaging.

---

### 2. Dataset Construction Diagram

**File:** `figures/dataset_construction.png`

**Place near:** the dataset or preprocessing section.

**Suggested content:** a diagram showing how one paper becomes a positive pair and how negative pairs are formed from different papers/domains.

---

### 3. Model Comparison Chart

**File:** `figures/model_comparison.png`

**Place near:** the model comparison or results section.

**Suggested content:** a bar chart comparing Accuracy, Macro-F1, ROC-AUC, and PR-AUC across all six models.

---

### 4. Confusion Matrix Grid

**File:** `figures/confusion_matrix_grid.png`

**Place near:** the results section.

**Suggested content:** one confusion matrix per model in a 2×3 grid.

---

### 5. ROC Curves Grid

**File:** `figures/roc_curves_grid.png`

**Place near:** the results section, after the confusion matrices.

**Suggested content:** overlayed or grid ROC curves for all models.

---

### 6. PR Curves Grid

**File:** `figures/pr_curves_grid.png`

**Place near:** the results section, after the ROC curves.

**Suggested content:** overlayed or grid precision–recall curves for all models.

---

### 7. Subdomain Performance Grid

**File:** `figures/subdomain_performance_grid.png`

**Place near:** the discussion section.

**Suggested content:** per-subdomain accuracy plots for each model.

---

### 8. Length Bucket Grid

**File:** `figures/length_bucket_grid.png`

**Place near:** the discussion section.

**Suggested content:** accuracy across short, medium, long, and very long input-length buckets.

---

## Recommended README Visual Order

A good visual order in GitHub is:

1. Pipeline overview
2. Dataset construction
3. Model comparison
4. Confusion matrices
5. ROC curves
6. PR curves
7. Subdomain performance
8. Length-bucket performance

This order helps the reader move from **how the system works** to **how well it performs**.

---

## Data Collection

The dataset is extracted from **PMC Open Access** using the official OAI-PMH interface.

### Extracted fields

* paper title
* abstract
* conclusion
* raw metadata tags
* MeSH-related tags
* source ID / PMCID

### Extraction logic

The parser is designed to handle variation in XML structure. For example, conclusion text may appear under headings such as:

* Conclusion
* Conclusions
* Summary and conclusion
* Final remarks

Only records with both a usable abstract and a usable conclusion are kept.

---

## MeSH and Subdomain Grouping

**MeSH** stands for **Medical Subject Headings**. It is a controlled biomedical vocabulary maintained by the National Library of Medicine. MeSH helps standardize topics across papers so that related terms map to broader subject groups.

In this project, MeSH-like tags are used to group papers into biomedical subdomains such as:

* Oncology
* Cardiology / Vascular
* Neurology
* Immunology / Infectious Disease
* Endocrinology / Metabolism
* Gastroenterology / Hepatology
* Nephrology / Urology
* Genetics / Molecular Biology
* Diagnostics / Imaging / Methods
* and others

These groups support subdomain-aware dataset construction and later analysis.

---

## Dataset Construction

The final dataset is **pair-level**.

### Positive pairs

A positive sample is created from the **abstract and conclusion of the same paper**.

### Negative pairs

A negative sample is created by pairing an abstract from one paper with a conclusion from a different paper, preferably from a different subdomain.

### Why this works

This creates a supervised binary task where the model learns whether two biomedical sections are semantically aligned within the same subdomain.

---

## Training Setup

The benchmark compares six models:

* **SBERT + MLP** — fast sentence-embedding baseline
* **BioBERT** — biomedical cross-encoder
* **Longformer** — long-context transformer
* **BigBird** — sparse long-context transformer
* **T5** — prompt-based text-to-text classifier
* **BART** — encoder-decoder classifier

All models are trained with a common evaluation protocol and threshold tuning.

### Common techniques

* class weighting
* label smoothing
* mixed precision
* gradient accumulation
* early stopping
* validation-based threshold optimization

---

## Evaluation Metrics

The project reports:

* Accuracy
* Macro Precision
* Macro Recall
* Macro F1
* Weighted F1
* ROC-AUC
* PR-AUC
* Log-Loss
* Confusion Matrix
* Subdomain-wise Performance
* Length-bucket Performance
* Hard-negative Accuracy

---

## How to Use the Project

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd ACRD-Project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If you are using Colab, install the packages in the notebook cells instead.

---

### 3. Prepare the data

The project expects the extracted dataset files in the `data/` folder.

Typical files:

* `pmc_abstract_conclusion_data.pkl`
* `all_unique_mesh_tags.csv`
* `pmc_pair_dataset_balanced.csv`

If you are rebuilding from scratch, run the extraction notebook first, then the dataset-building notebook.

---

### 4. Build the pair-level dataset

Run the dataset construction notebook to:

* clean the text
* assign subdomains
* generate positive and negative pairs
* save the final CSV file

The final dataset should contain the following minimum columns:

* `abstract_clean`
* `conclusion_clean`
* `label`
* `abstract_subdomain`
* `conclusion_subdomain`
* `abs_wc`
* `concl_wc`

---

### 5. Train the models

Run the training notebook one model at a time.

Each model saves its own directory under `saved_models/`.

Example output structure:

```text
saved_models/BART/
├── model/
├── tokenizer/
├── results/
├── best_model.pt
├── metrics.json
├── threshold.json
├── label_map.json
├── inference.py
└── training_config.json
```

---

### 6. Compare results

After training, inspect:

* metrics in `metrics.json`
* plots in `results/`
* model comparison figures in `figures/`

---

### 7. Run inference

Use the saved inference script for the model you want to deploy.

For example, for BART:

```bash
python saved_models/BART/inference.py
```

You can also import the inference function and pass a custom abstract/conclusion pair.

---

## BART Inference Example

The BART inference script supports multiline abstract and conclusion input and returns:

* `label`
* `label_name`
* `probability_same_subdomain`
* `threshold`

### Example output

```python
{
    "label": 1,
    "label_name": "Same subdomain",
    "probability_same_subdomain": 0.9734,
    "threshold": 0.87
}
```

---

## Reproducibility

To reproduce the experiments:

* keep the random seed fixed at `42`
* use the same dataset split files
* keep the same preprocessing steps
* use the same threshold tuning procedure
* load the same saved checkpoints

---

## Limitations

* Performance depends on the quality of subdomain grouping.
* Some biomedical papers span multiple domains.
* Long-context models are slower and more memory intensive.
* Prompt-based models may be sensitive to phrasing.

---

## Future Work

Possible extensions include:

* paper-level split enforcement
* stronger MeSH normalization
* external validation on PubMed-only corpora
* hierarchical subdomain classification
* explanation generation for predicted labels
* deployment as a web app or local API

---

## Citation

If you use this project in your work, cite the relevant biomedical and transformer model references used in the study.

---

## Contact

Add your name, email, or GitHub profile here.

---

## License

Add your project license here, for example:

* MIT
* Apache 2.0
* GNU GPL v3

---

## Acknowledgements

* PMC Open Access for full-text biomedical articles
* NCBI / NLM for MeSH and biomedical indexing resources
* Hugging Face Transformers and Sentence-Transformers for model tooling
