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

## Project Structure

```text
ACRD-Project/
├── notebooks/
│   ├── Dataset_Generation.ipynb
│   ├── All_Models.ipynb
├── saved_models/
│   ├── SBERT/
│   ├── BioBERT/
│   ├── Longformer/
│   ├── BigBird/
│   ├── T5/
│   └── BART/
└── README.md
```

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

## Dataset Construction

The final dataset is **pair-level**.

### Positive pairs

A positive sample is created from the **abstract and conclusion of the same paper**.

### Negative pairs

A negative sample is created by pairing an abstract from one paper with a conclusion from a different paper, preferably from a different subdomain.

### Why this works

This creates a supervised binary task where the model learns whether two biomedical sections are semantically aligned within the same subdomain.

## Models and Results

This project benchmarks six deep learning models for biomedical abstract–conclusion relevance detection. Every model is trained on the same pair-level dataset, uses the same label definition, and is evaluated using the same validation-driven threshold selection procedure. The goal is not only to compare final accuracy, but also to compare how different architectural families behave on biomedical text pairs under identical preprocessing and evaluation settings.

### 1. SBERT + MLP

SBERT is the fastest model in the benchmark and serves as a strong embedding-based baseline. It encodes the abstract and conclusion independently into dense sentence representations. Those representations are then combined with pairwise interaction features and passed through a lightweight multilayer perceptron. Because the encoder is frozen and only the classifier head is trained, SBERT is highly efficient and inexpensive to run. In this project, SBERT is especially useful as a practical reference model: it shows how far a sentence-embedding approach can go before moving to heavier transformer cross-encoders.

### 2. BioBERT

BioBERT is a biomedical transformer pretrained on PubMed and PMC-style text. It is used here as a cross-encoder, meaning the abstract and conclusion are concatenated and processed jointly. This is a stronger formulation than independent encoding because the model can attend across the two sections token by token. BioBERT is expected to perform well on this task because the dataset is biomedical, the labels are subdomain-based, and the language is technical and domain-specific. Its strong results confirm that biomedical pretraining is highly valuable for this classification problem.

### 3. Longformer

Longformer is designed for long documents and replaces standard dense self-attention with a sparse attention pattern. This makes it suitable for longer biomedical inputs, where important information may be distributed across many tokens. In the benchmark, Longformer is used as a pair classifier on abstract and conclusion text. It is particularly relevant when the input text is longer than what standard BERT-style models can handle comfortably. Although Longformer remains strong, its training cost is higher than simpler models, and in this setup it is outperformed by the top models in overall classification quality.

### 4. BigBird

BigBird is another long-context transformer, but it uses a block-sparse attention design that mixes local, global, and random attention. This allows it to process long sequences efficiently while still capturing long-range dependencies. For this project, BigBird is important because biomedical abstracts and conclusions may contain distant clues that matter for subdomain classification. BigBird performs extremely well in the benchmark and achieves the strongest ranking-style metrics, which suggests that its sparse attention mechanism captures the pair-level semantic relationship very effectively.

### 5. T5

T5 is a text-to-text model adapted here for prompt-based classification. Instead of using a traditional classification head, the model receives a natural-language prompt and predicts a label through token likelihood scoring. This makes T5 flexible, but also more sensitive to prompt design and calibration. In this project, T5 is useful as a contrasting architecture because it tests whether the task can be solved well using a text-to-text framing rather than a direct classification head. The results show that T5 is still capable of learning the task, but it is less competitive than the strongest encoder and encoder-decoder classifiers under the current setup.

### 6. BART

BART is an encoder-decoder model trained using a denoising objective. For this project, it is used in sequence-classification mode, which lets the encoder process the abstract and conclusion jointly and pass the pooled representation to a classification head. BART is one of the strongest models in the benchmark and achieves the best overall classification performance. This suggests that its combination of a robust pretrained encoder and a strong classification head is particularly effective for biomedical pair classification.

### Test Results Summary

The following table summarizes the test performance of the six models. The values show that all models perform well, but there are clear differences in accuracy, calibration, and training efficiency.

| Model       | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Weighted F1 | ROC-AUC | PR-AUC | Log-Loss |
| ----------- | -------: | ----------------: | -------------: | ---------: | ----------: | ------: | -----: | -------: |
| SBERT + MLP |   0.9941 |            0.9935 |         0.9935 |     0.9935 |      0.9941 |  0.9998 | 0.9996 |   0.0237 |
| BioBERT     |   0.9946 |            0.9927 |         0.9955 |     0.9941 |      0.9946 |  0.9999 | 0.9997 |   0.0212 |
| Longformer  |   0.9833 |            0.9819 |         0.9813 |     0.9816 |      0.9832 |  0.9981 | 0.9971 |   0.0608 |
| BigBird     |   0.9956 |            0.9959 |         0.9943 |     0.9951 |      0.9956 |  0.9999 | 0.9998 |   0.0234 |
| T5          |   0.9591 |            0.9581 |         0.9516 |     0.9547 |      0.9590 |  0.9932 | 0.9860 |   0.1943 |
| BART        |   0.9970 |            0.9971 |         0.9964 |     0.9967 |      0.9970 |  0.9993 | 0.9988 |   0.0194 |

### Key Observations

The benchmark shows a clear but narrow separation among the top models. **BART** achieves the best overall classification performance, with the highest accuracy and macro-F1. **BigBird** follows very closely and obtains the strongest ROC-AUC and PR-AUC, which indicates excellent ranking quality. **BioBERT** also performs extremely well, showing that domain-specific biomedical pretraining is very effective for this task.

**SBERT** is the most efficient model in the benchmark and remains highly competitive despite being much simpler than the transformer-based cross-encoders. This makes it the best model when training time or deployment cost is a concern. **Longformer** still performs strongly, but its scores are lower than those of the top three models. **T5** is the weakest model in the comparison, likely because prompt-based label-token scoring is more sensitive to phrasing, calibration, and optimization choices.

### Training Time

| Model       | Training Time (s) |
| ----------- | ----------------: |
| SBERT + MLP |              29.4 |
| BioBERT     |             441.7 |
| Longformer  |            3026.6 |
| BigBird     |            3311.8 |
| T5          |            5111.0 |
| BART        |            2360.6 |

The training-time table shows an important practical trade-off. SBERT is by far the fastest model, while T5 is the slowest. Longformer and BigBird are computationally expensive because of long-context processing, and BART offers a strong balance between high accuracy and moderate training time.

### Best Validation Thresholds

Each model uses a validation-optimized threshold instead of a fixed cutoff of 0.5. This is important because the raw probability distributions differ across architectures and are not equally calibrated.

| Model       | Best Threshold |
| ----------- | -------------: |
| SBERT + MLP |          0.765 |
| BioBERT     |          0.105 |
| Longformer  |          0.195 |
| BigBird     |          0.050 |
| T5          |          0.880 |
| BART        |          0.870 |

The threshold values show that the models behave very differently in probability space. Some models are conservative and require a high cutoff for positive predictions, while others produce high positive probabilities more readily. Validation-based threshold tuning therefore plays a central role in obtaining the best final test performance.

### Interpretation

Overall, the benchmark indicates that abstract–conclusion relevance detection is highly learnable when the dataset is carefully constructed and the models are trained with class weighting, validation threshold tuning, and consistent preprocessing. The best model depends on the intended use case. If the priority is maximum predictive performance, **BART** is the strongest option. If the priority is ranking quality on difficult examples, **BigBird** is especially strong. If the priority is biomedical specialization, **BioBERT** is a very solid choice. If the priority is speed and simplicity, **SBERT** offers the best speed–performance trade-off.

### Saved Outputs

Each trained model saves the following artifacts in its own directory:

* `best_model.pt`
* `metrics.json`
* `threshold.json`
* `label_map.json`
* `training_config.json`
* `inference.py`
* `model/`
* `tokenizer/`
* `results/`

These files can be used later for inference, packaging, comparison, or deployment without retraining.

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

## How to Use the Project

### Clone the repository

```bash
git clone <your-repo-url>
cd ACRD-Project
```

### Install dependencies

Install the packages in the notebook cells.

## Run BART inference cell

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

## Acknowledgements

* PMC Open Access for full-text biomedical articles
* NCBI / NLM for MeSH and biomedical indexing resources
* Hugging Face Transformers and Sentence-Transformers for model tooling
