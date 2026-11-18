# Toxic Comment Detection
### Multi-Label Text Classification with NLP & Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-4EAA25?style=flat-square)](https://www.nltk.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

**Live app:** https://toxic-comment-detection-01.streamlit.app/

---

## Overview

Online platforms have to moderate harmful content at scale. This project builds a **multi-label toxic comment classifier**
that detects six types of harmful language at the same time, based on the
[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).

A comment can belong to **several categories at once**, for example both *obscene* and *insulting*.
The **Streamlit app** scores any comment you type across all six labels.

---

## Toxicity Categories

| # | Label | Description |
|---|---|---|
| 1 | `toxic` | Generally offensive or harmful content |
| 2 | `severe_toxic` | Extremely offensive content |
| 3 | `obscene` | Explicit or obscene language |
| 4 | `threat` | Threats of violence or harm |
| 5 | `insult` | Personal insults |
| 6 | `identity_hate` | Hate speech targeting identity groups |

---

## Methodology

```
Raw comment text
      │
      ▼
 Cleaning
  ├─ Drop malformed rows (labels must be 0/1)
  ├─ Lowercase, remove punctuation & digits
  ├─ Lemmatisation (WordNet) + stemming (Porter)
  └─ Stop word removal, comments up to 400 characters
      │
      ▼
 Stratified train / test split (75 / 25)   ← vectorisers are fitted on training text only (no leakage)
      │
      ▼
 Features
  ├─ Bag-of-Words (CountVectorizer, 10k terms)
  ├─ TF-IDF (10k terms)
  └─ TF-IDF with bigrams (20k terms)
      │
      ▼
 Models
  ├─ Binary Relevance: Multinomial Naïve Bayes        ← BoW
  ├─ Binary Relevance: Linear SVM, class_weight       ← TF-IDF
  ├─ Binary Relevance: Linear SVM + SMOTE             ← TF-IDF
  ├─ Classifier Chain: Linear SVM                     ← TF-IDF + bigrams   (deployed)
  └─ Binary Relevance: Linear SVM tuned (GridSearchCV)
      │
      ▼
 Evaluation: Hamming loss, exact match accuracy, macro F1, per-label F1 and confusion matrices
```

---

## Results

Test set of 28,692 comments.

| Model | Hamming Loss ↓ | Exact Match ↑ | Macro F1 ↑ |
|---|---|---|---|
| BR - Multinomial Naïve Bayes (BoW) | 3.67% | 87.1% | 0.494 |
| **BR - Linear SVM, class-balanced (TF-IDF)** | **3.24%** | 86.2% | **0.567** |
| BR - Linear SVM + SMOTE (TF-IDF) | 5.03% | - | 0.470 |
| Classifier Chain - Linear SVM (TF-IDF + bigrams) | 3.37% | **87.3%** | 0.562 |
| BR - Linear SVM tuned (GridSearchCV) | 3.24% | 86.8% | 0.566 |

**Per-label F1 (best model per label):** Toxic 0.77 · Obscene 0.81 · Insult 0.69 · Severe Toxic 0.43 · Threat 0.41 · Identity Hate 0.40

**Findings**
- **Class weighting** helps clearly: Linear SVM with `class_weight='balanced'` beats Naïve Bayes (macro F1 0.567 vs 0.494).
- **SMOTE hurts** on sparse TF-IDF text (0.470): interpolated text vectors are unrealistic.
- **Classifier Chains** reach the best exact-match accuracy and almost the same macro F1 as the best model, so the chain is deployed in the app.
- **Tuning** (grid search on the Toxic label) gives no extra gain. Its `C = 0.1` hurts the chain, so the chain keeps `C = 1.0`.
- **Rare labels** (threat, identity hate, severe toxic) are the hardest, with only a few hundred examples each.

---

## Streamlit App

- Type or paste a comment, press **ANALYSE**, and see which labels apply and a score for each.
- Words the model has never seen are treated as clean instead of guessed.
- The app loads `model.pkl` and `vectoriser.pkl` from the repo. Without them it falls back to a small demo model.

Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure

```
toxic-comment-detection/
├── Toxic Comment Detection.ipynb   ← full analysis and model training
├── app.py                          ← Streamlit app
├── model.pkl                       ← trained Classifier Chain (created by the notebook)
├── vectoriser.pkl                  ← fitted TF-IDF vectoriser (created by the notebook)
├── requirements.txt                ← scikit-learn pinned to the version the model was saved with
├── README.md
└── LICENSE
```

The dataset (`ToxicDataSet.csv`) is not included. Download it from the Kaggle competition above.

---

## Future Improvements

| Priority | Improvement | Expected Impact |
|---|---|---|
| High | Fine-tune **BERT / DistilBERT** | Contextual embeddings, largest potential gain |
| High | **Per-label decision thresholds** | Better precision / recall balance on rare labels |
| Medium | **Calibrated probabilities** (`CalibratedClassifierCV`) | Meaningful confidence scores in the app |
| Medium | Tune `C` **per label** instead of on Toxic only | Better rare-label F1 |
| Low | Error analysis on false positives / negatives | Understand remaining failure modes |

---

## References

- Jigsaw / Conversation AI. *Toxic Comment Classification Challenge.* Kaggle, 2018.
- Read, J. et al. (2011). *Classifier Chains for Multi-Label Classification.* Machine Learning, 85(3).
- Zhang, M. L., & Zhou, Z. H. (2014). *A Review on Multi-Label Learning Algorithms.* IEEE TKDE.
- Chawla, N.V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR, 16.

---

## License

This project is licensed under the [MIT License](LICENSE).
