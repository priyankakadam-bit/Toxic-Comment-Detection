# Toxic Comment Detection
### Multi-Label Text Classification with NLP & Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-4EAA25?style=flat-square)](https://www.nltk.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

---

## Overview

Online platforms face the constant challenge of moderating harmful content at scale. This project builds a **multi-label toxic comment classifier** that simultaneously detects six types of harmful language in user-generated text — replicating the core task of the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).

Unlike single-label classification, a comment can belong to **multiple categories at once** — for example, both *obscene* and *insulting* at the same time.

A fully interactive **Streamlit web app** is included so you can test the classifier on any comment in real time.

---

## Toxicity Categories

| # | Label | Description |
|---|---|---|
| 1 | `toxic` | Generally offensive or harmful content |
| 2 | `severe_toxic` | Extremely offensive or threatening content |
| 3 | `obscene` | Explicit or sexually obscene language |
| 4 | `threat` | Direct threats of violence or harm |
| 5 | `insult` | Personal insults targeting individuals |
| 6 | `identity_hate` | Hate speech targeting identity groups |

---

## Methodology

```
Raw Comment Text
      │
      ▼
 Preprocessing
  ├─ Lowercase
  ├─ Remove punctuation & digits
  ├─ Lemmatisation (WordNet)
  ├─ Stemming (Porter)
  └─ Stop word removal
      │
      ▼
 Feature Extraction
  ├─ Bag-of-Words  (CountVectorizer)
  └─ TF-IDF with bigrams (TfidfVectorizer, ngram_range=(1,2))
      │
      ▼
 Models Compared (Binary Relevance strategy)
  ├─ Model 1: Multinomial Naïve Bayes         ← BoW features
  ├─ Model 2: Linear SVM + class weighting    ← TF-IDF features
  ├─ Model 3: Linear SVM + SMOTE              ← TF-IDF features
  ├─ Model 4: Classifier Chains (SVM)         ← TF-IDF + bigrams   Best
  └─ Model 5: SVM + GridSearchCV tuning       ← TF-IDF + bigrams
      │
      ▼
 Evaluation
  ├─ Hamming Loss
  ├─ Exact Match Accuracy
  ├─ Macro F1-Score
  └─ Per-label Confusion Matrices
```

---

## Results

| Model | Hamming Loss ↓ | Exact Match ↑ | Macro F1 ↑ |
|---|---|---|---|
| BR — Multinomial Naïve Bayes (BoW) | ~3.69% | — | — |
| BR — Linear SVM, balanced (TF-IDF) | ~2.88% | — | — |
| BR — Linear SVM + SMOTE | — | — | — |
| **Classifier Chain (TF-IDF + bigrams)** | **Best** | **Best** | **Best** |
| BR — SVM Tuned (GridSearchCV) | — | — | — |

> Exact results vary slightly with each run due to random sampling. The **Classifier Chain** consistently outperforms Binary Relevance models by capturing label dependencies (e.g. a `severe_toxic` comment is almost always also `toxic`).

---

## Project Structure

```
Toxic-Comment-Detection/
├──  Toxic Comment Detection.ipynb   ← full analysis & model training
├──  app.py                          ← Streamlit demo app
├──  README.md
├──  requirements.txt
├──  LICENSE
└──  .gitignore
```

> `ToxicDataSet.csv`, `model.pkl`, and `vectoriser.pkl` are excluded from the repo (see `.gitignore`). Generate the model files by running the notebook.

---

## Requirements

```
pandas>=1.3
numpy>=1.21
matplotlib>=3.5
seaborn>=0.12
scikit-learn>=1.0
nltk>=3.7
ydata-profiling>=4.0
scipy>=1.7
streamlit>=1.30
joblib>=1.3
imbalanced-learn>=0.11
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## Future Improvements

| Priority | Improvement | Expected Impact |
|---|---|---|
| 🔴 High | Fine-tune **BERT / RoBERTa** | Large accuracy gains via contextual embeddings |
| 🔴 High | **SMOTE + class weighting** combined | Better recall on rare labels |
| 🟡 Medium | **Classifier Chains** with optimised order | Capture label dependencies more effectively |
| 🟡 Medium | **Hyperparameter tuning** | Optimise C, alpha, feature count |
| 🟢 Low | **Trigram features** | Marginal gains over bigrams |
| 🟢 Low | **Error analysis on FP/FN** | Understand remaining failure modes |

---

## References

- Jigsaw / Conversation AI. *Toxic Comment Classification Challenge.* Kaggle, 2018.
- Read, J. et al. (2011). *Classifier Chains for Multi-Label Classification.* Machine Learning, 85(3).
- Zhang, M. L., & Zhou, Z. H. (2014). *A Review on Multi-Label Learning Algorithms.* IEEE TKDE.
- Chawla, N.V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR, 16.

---

## License

This project is licensed under the [MIT License](LICENSE).
