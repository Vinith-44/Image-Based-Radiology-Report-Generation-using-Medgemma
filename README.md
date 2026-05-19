# Image-Based Radiology Report Generation using MedGemma

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/vinithvanjangi/radiology-image-based-report-generation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RADAR (Multi-modal Evidence-anchored Radiology Report Generation)** is an advanced AI pipeline designed to generate clinical-grade, hallucination-free radiology reports from Chest X-rays, CT scans, and MRI scans. Built around Google's MedGemma 4B, the system utilizes a custom Retrieval-Augmented Generation (RAG) framework to ensure all generated findings are strictly grounded in retrieved clinical evidence.

---

## Table of Contents
- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Datasets](#-datasets)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage & Quick Start](#-usage--quick-start)
- [Evaluation Metrics](#-evaluation-metrics)
- [Team & Contributors](#-team--contributors)

---

## Overview

Radiologists face immense workloads reviewing thousands of imaging studies daily. RADAR automates the initial drafting of structured radiology reports (Technique, Findings, Impression). Unlike single-modality models, RADAR handles X-ray, CT, and MRI by extracting hybrid semantic features and leveraging a 3-tier indexing RAPTOR system to pull exact clinical patterns before generation.

---

## System Architecture

The project is divided into four highly specialized modules:

1. **GHFE (Graph-Guided Hybrid Feature Extractor)**
   - **Backbone:** Unfrozen `DenseNet-121` (denseblock 3 & 4) for visual feature extraction.
   - **Mechanism:** Uses a vectorized 2-layer GCN over modality-specific knowledge graphs (40-node X-Ray, 38-node CT, 35-node MRI) to output disease probability arrays.

2. **RAMT (Robust Radiological Attention Mean Teacher)**
   - **Learning:** Semi-supervised Student-Teacher framework with Exponential Moving Average (EMA).
   - **Preprocessing:** Applies modality-aware CLAHE (Contrast Limited Adaptive Histogram Equalization).
   - **Loss Functions:** Binary Focal Loss (for labeled NIH data) + Cosine Consistency Loss.

3. **RadioShield RAG (Hallucination-Free Radiology RAG)**
   - **Retrieval:** Multi-Channel Retrieval (BM25, FAISS HNSW, HyDE, Graph GHFE) using Reciprocal Rank Fusion (RRF).
   - **Reranking:** Cross-Encoder (`ms-marco-MiniLM`) with min-max normalized Maximal Marginal Relevance (MMR).
   - **Guardrails:** NLI Faithfulness Guard filters out clinical hallucinations and contradictions prior to prompt injection.

4. **MedGemma Generation**
   - **LLM:** Google MedGemma 4B.
   - **Prompt Engineering:** CEMRAG concept context + Negative Constraints Injection (e.g., forcing the model *not* to mention CT Hounsfield units when evaluating an X-ray).

---

## Datasets

The system trains and retrieves context from the following medical imaging datasets:
* **Chest X-Ray:** MIMIC-CXR, Indiana University CXR, NIH ChestX-ray14
* **CT Scans:** CT-RATE, Brain Stroke CT
* **MRI Scans:** Augmented Alzheimer MRI, Brain Tumor Classification MRI

*Note: Due to privacy (HIPAA) and size constraints, raw images are not hosted in this repository.*

---

## Project Structure

```text
Image-Based-Radiology-Report-Generation-using-Medgemma/
├── data/
│   └── Datasets.md          # Dataset download links
├── models/
│   └── Models.md            # Model weights download link (HuggingFace)
├── notebooks/
│   └── radiology-image-based-report-generation.ipynb
├── outputs/
│   └── report.pdf
├── src/
│   ├── ghfe_module.py
│   ├── medgemma_engine.py
│   ├── medgemma_report_generator.py
│   ├── radioshield_rag.py
│   └── ramt_module.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Vinith-44/Image-Based-Radiology-Report-Generation-using-Medgemma.git
cd Image-Based-Radiology-Report-Generation-using-Medgemma
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

*Core dependencies:* `torch`, `transformers`, `sentence-transformers`, `faiss-cpu`, `rank_bm25`, `nltk`, `opencv-python`.

---

##  Usage & Quick Start

### 1. Training the RAMT Module

To train the semi-supervised feature extractor on your dataset (ensure `all_train.json` is in the `data/` folder):

```bash
python src/ramt_module.py
```

*This will output a training dashboard (`outputs/ramt_v2_training_dashboard.png`) and save the best models to the `models/` directory.*

### 2. Running the RadioShield RAG Pipeline

You can test the RAG retrieval and prompt generation directly using the built-in demo logic:

```python
from src.radioshield_rag import RadioShieldRAG

# Initialize and build the multi-channel FAISS/BM25 index
rag = RadioShieldRAG()
rag.build_knowledge_base(use_saved_index=False)

# Simulate a CT scan query with GHFE probabilities
ct_probs = {"pulmonary_embolism": 0.83, "lung_consolidation_ct": 0.34}

# Generate the prompt augmented with retrieved evidence and constraints
enriched_prompt, rag_out = rag.get_medgemma_prompt(
    query="45-year-old post-surgical patient with sudden dyspnea and tachycardia",
    modality="ct",
    disease_probs=ct_probs
)

# Print the RAG Pipeline execution report including NLI hallucination checks
rag.print_rag_report(rag_out)
```

### 3. Full Pipeline Notebook

The end-to-end pipeline (preprocessing → RAMT → RAG → MedGemma generation) is available as a Kaggle notebook:

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/vinithvanjangi/radiology-image-based-report-generation)

---

## Evaluation Metrics

The pipeline's generated text is rigorously evaluated against ground-truth clinical reports (e.g., Indiana CXR + MIMIC-CXR). Baseline metrics:

| Metric | Score | Description |
|--------|-------|-------------|
| **BLEU-1** | 0.6134 | Vocabulary precision and unigram overlap |
| **BLEU-4** | 0.2618 | Phrase-level accuracy in generated clinical text |
| **ROUGE-L** | 0.5102 | Structural report similarity via longest common subsequence |
| **METEOR** | 0.6472 | Semantic similarity with synonym matching |

**RAGAS** is also used as a dedicated pipeline measuring *Faithfulness*, *Answer Relevancy*, *Context Recall*, and tracking strict hallucination/contradiction rates.

---

## Team & Contributors

**Institution:** B.V. Raju Institute of Technology (BVRIT), Narsapur, Medak, India.

**Department:** Computer Science and Engineering

| Name | Roll Number |
|------|-------------|
| **Vinith** | 24211A05KZ |
| **Udayana Ram Kiran** | 24211A05KN |
| **Sowmy Raj Singh** | 24211A05JK |
| **Ch. Vivek Karthik** | 24211A05LK |

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
