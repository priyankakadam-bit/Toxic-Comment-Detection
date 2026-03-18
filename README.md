# Toxic Comment Detection — Multi-Label Text Classification

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A multi-label text classification system that detects **six types of online toxicity** in user comments, using classical NLP and machine learning techniques.

---

## Overview

Each comment can simultaneously belong to one or more of the following toxicity categories:

| Label | Description |
|---|---|
| `toxic` | Generally offensive or harmful content |
| `severe_toxic` | Extremely offensive or threatening content |
| `obscene` | Explicit or sexually obscene language |
| `threat` | Direct threats of violence or harm |
| `insult` | Personal insults or targeted demeaning language |
| `identity_hate` | Hate speech targeting identity groups |

---

## Project Structure

```
toxic-comment-detection/
├── Toxic_Comment_Detection.ipynb   ← Main notebook
├── ToxicDataSet.csv                ← Dataset (download from Kaggle)
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/<your-username>/toxic-comment-detection.git
cd toxic-comment-detection

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle:
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
# Place train.csv in the project folder and rename it ToxicDataSet.csv

# Launch the notebook
jupyter notebook Toxic_Comment_Detection.ipynb
```

---

## Pipeline

```
Raw Text
   │
   ▼
Lowercase + Remove Punctuation / Digits
   │
   ▼
Stemming + Lemmatization (NLTK)
   │
   ▼
Stop Word Removal
   │
   ▼
Bag-of-Words Vectorisation (CountVectorizer)
   │
   ▼
Binary Relevance — one classifier per label
   ├──► Multinomial Naïve Bayes
   └──► Linear SVM
```

---

## Results

| Model | Hamming Loss | Exact Match Accuracy |
|---|---|---|
| BR — Multinomial Naïve Bayes | ~3.69% | — |
| BR — Support Vector Machine  | ~2.88% | — |

> **Lower Hamming Loss = fewer label prediction errors.**  
> The SVM outperforms Naïve Bayes on this high-dimensional, sparse text data.

---

## Future Improvements

- Replace raw counts with **TF-IDF** weighting
- Use **Classifier Chains** to capture label dependencies
- Fine-tune a **BERT** model for richer semantic representation
- Apply **SMOTE** or class weighting to handle the class imbalance
- Hyperparameter tuning via `GridSearchCV`

---

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
scikit-multilearn
nltk
ydata-profiling
scipy
```

---

## Dataset

- **Source:** [Jigsaw Toxic Comment Classification Challenge (Kaggle)](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- ~162,000 Wikipedia comments labelled by human raters for toxicity

---

## License

This project is licensed under the [MIT License](LICENSE).
