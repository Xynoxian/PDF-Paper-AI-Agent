# D1 — Streaming Learner & AutoML Note

Auto-tuned hybrid kNN retriever (Optuna) + River online learner with ADWIN drift detection, evaluated over an arXiv scientific papers corpus.

---

## Table of contents
- [Project overview](#project-overview)
- [Repo structure](#repo-structure)
- [Quickstart — Google Colab](#quickstart--google-colab)
- [Kaggle API setup (secure — no file sharing)](#kaggle-api-setup-secure--no-file-sharing)
- [Running the notebook](#running-the-notebook)
- [Outputs](#outputs)
- [Run card](#run-card)
- [Dependencies](#dependencies)

---

## Project overview

This deliverable implements two components on top of an arXiv paper corpus:

| Component | What it does |
|---|---|
| **AutoML (Track A)** | Optuna TPE search over k, alpha, SVD dim, norm, and metric for a hybrid BM25 + dense retriever |
| **Online learner** | River logistic regression that adapts the hybrid fusion weight (alpha) from user feedback in real time |
| **Drift detection** | ADWIN monitors the feedback stream and fires when the error rate shifts significantly |
| **Evaluation** | NDCG@5, Recall@5, p95 latency — baseline vs AutoML best |

---

## Repo structure

```
d1/
├── d1_automl.ipynb        # Main notebook (run this)
├── run_card.yaml          # Winning config + all metrics (auto-generated)
├── metrics_summary.csv    # Baseline vs AutoML table (auto-generated)
├── automl_search.png      # Optuna convergence plot (auto-generated)
├── prequential_chart.png  # River + ADWIN drift chart (auto-generated)
└── README.md              # This file
```

---

## Quickstart — Google Colab

> **Recommended environment.** Colab provides free GPU/CPU and lets teammates share a single notebook link. No local Python setup needed.

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook** and upload `d1_automl.ipynb`
3. Follow the Kaggle setup steps below **before** running any cells
4. Run cells top to bottom (**Runtime → Run all**)

---

## Kaggle API setup (secure — no file sharing)

> ⚠️ **Never commit or share your `kaggle.json` file.** It contains your private API key. The method below stores it securely inside Colab's encrypted Secrets vault — it never touches the repo.

### Step 1 — Get your Kaggle API key

1. Sign in at [kaggle.com](https://kaggle.com)
2. Click your profile picture → **Settings**
3. Scroll to the **API** section → click **Create New Token**
4. This downloads a file called `kaggle.json` to your computer. Open it — it looks like:
   ```json
   {"username": "your_username", "key": "your_api_key_here"}
   ```
5. Note down the two values (`username` and `key`) — you will need them in the next step

### Step 2 — Add your credentials to Colab Secrets

Colab Secrets is an encrypted store built into Colab. Your key is tied to your Google account and is never visible to anyone else.

1. In your Colab notebook, click the **🔑 key icon** in the left sidebar (or go to **Tools → Secrets**)
2. Click **+ Add new secret** and add these two entries:

   | Name | Value |
   |---|---|
   | `KAGGLE_USERNAME` | your Kaggle username |
   | `KAGGLE_KEY` | the key string from your kaggle.json |

3. Toggle **Notebook access** to ON for both secrets

### Step 3 — The notebook reads them automatically

Cell 2 of the notebook contains the following code that pulls the secrets and configures Kaggle without ever writing your key to disk:

```python
from google.colab import userdata
import os

os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"]      = userdata.get("KAGGLE_KEY")
```

> If you are running **locally** (not Colab), place your `kaggle.json` at `~/.kaggle/kaggle.json` and set permissions with `chmod 600 ~/.kaggle/kaggle.json`. Add `kaggle.json` to your `.gitignore` — **do not commit it**.

---

## Running the notebook

Once Kaggle secrets are configured, run cells in order:

| Cell | What happens |
|---|---|
| **0** | Installs all dependencies |
| **1** | Imports libraries, sets global config |
| **2** | Downloads arXiv dataset via Kaggle, filters to AI/ML papers, builds corpus |
| **3** | Encodes all abstracts + query titles into dense vectors (bge-small-en) |
| **4** | Builds BM25 keyword index |
| **5** | Defines `HybridRetriever` class |
| **6** | Defines NDCG@5 / Recall@5 / latency evaluation helpers |
| **6b** | Runs baseline evaluation (default params) |
| **7** | Runs Optuna AutoML search — 40 trials, saves best config |
| **8** | Plots Optuna convergence, NDCG per trial, latency per trial |
| **9** | Runs River online learner with ADWIN drift simulation |
| **10** | Plots prequential accuracy chart with drift markers |
| **11** | Prints and saves metrics summary table |
| **12** | Saves `run_card.yaml` with winning config and all metrics |

**Expected total runtime:** ~8–15 minutes on Colab CPU (most time is spent in Cell 3 encoding embeddings).

---

## Outputs

All outputs are auto-generated when the notebook finishes. Check your Colab file browser (left sidebar → folder icon) to download them.

| File | Description |
|---|---|
| `run_card.yaml` | Full reproducible config — winning hyperparameters, metrics, random seed |
| `metrics_summary.csv` | Side-by-side baseline vs AutoML NDCG@5, Recall@5, p95 latency |
| `automl_search.png` | Three-panel Optuna plot (objective, NDCG, latency over 40 trials) |
| `prequential_chart.png` | River accuracy over 300 feedback steps with red ADWIN drift markers |

---

## Run card

The `run_card.yaml` is auto-generated at the end of the notebook. It records the exact configuration needed to reproduce the best result:

```yaml
run_card:
  experiment: D1-AutoML-kNN-Optuna
  seed: 42
  n_trials: 40
  embed_model: BAAI/bge-small-en-v1.5

winning_config:
  k: <auto-filled>
  alpha: <auto-filled>
  svd_dim: <auto-filled>
  norm: <auto-filled>
  metric: <auto-filled>

metrics:
  baseline:
    NDCG@5: <auto-filled>
    Recall@5: <auto-filled>
    p95_latency_ms: <auto-filled>
  automl_best:
    NDCG@5: <auto-filled>
    Recall@5: <auto-filled>
    p95_latency_ms: <auto-filled>

online_learning:
  library: River
  model: LogisticRegression (SGD, l2=1e-4)
  drift: ADWIN (delta=0.002)
```

---

## Dependencies

All installed automatically by Cell 0 in the notebook:

```
optuna
rank_bm25
sentence-transformers
scikit-learn
river
matplotlib
pyyaml
tqdm
kagglehub
```
