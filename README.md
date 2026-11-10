# Cross-Project Story Point Estimation Using SBERT

**Group 13 — NLP Project**

## Group Members

| No. | Name | Roll Number |
|:---:|---|---|
| 1 | Nidhal Ahmed | B230458CS |
| 2 | Shresth Tiwari | B230090CS |
| 3 | Raghuveer Singh | B230515CS |
| 4 | Niraj Kumar | B230463CS |
| 5 | Pankaj Singh Chauhan |  |

## Problem Statement

Automated story point estimation using issue descriptions across multiple software projects. The objective is to evaluate SBERT-based sentence embeddings under a cross-project random-split experimental protocol.

## Objectives

- Generate semantic representations of software issues using SBERT.
- Train model for story point regression.
- Evaluate predictive performance using MAE and MdAE.
- Assess the effectiveness of SBERT embeddings for cross-project estimation.

## Dataset

The dataset contains historical software issues collected from multiple projects. Each record includes:

- Issue ID and project information.
- Issue description (text).
- Assigned story points (target variable).

The dataset is preprocessed and divided according to the assigned random-split protocol.

## Setup

**Requirements:** Python 3.x

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running Experiments

Execute the following files in the specified order:

```bash
python 01_load_data.py
python 02_prepare_data.py
python 03_encode_text.py
```

Then execute the notebook:

```bash
jupyter notebook nlp-model-2.ipynb
```

Finally, run the evaluation script:

```bash
python 05_evaluate_models.py
```

**Execution order:**

1. `01_load_data.py` — Load the raw dataset.
2. `02_prepare_data.py` — Preprocess data and generate splits.
3. `03_encode_text.py` — Generate SBERT sentence embeddings.
4. `nlp-model-2.ipynb` — Train the prediction model.
5. `05_evaluate_models.py` — Evaluate model predictions.

Ensure that the required dataset files are available before execution and that each step completes successfully before proceeding to the next.