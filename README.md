# Sports vs Politics Text Classification

## 📌 Project Overview

This project focuses on binary text classification of news articles into **Sports** and **Politics** categories using classical Machine Learning techniques.

The objective is to compare model performance across two datasets with different characteristics:

- BBC News Dataset (clean and curated)
- AG News Dataset (larger and more realistic)

---

## 📊 Datasets Used

### 1️⃣ BBC Dataset
- 928 total articles
- 511 Sports
- 417 Politics
- Clean and highly separable

### 2️⃣ AG News Dataset
- Filtered classes:
  - World → Politics
  - Sports → Sports
- ~7,600 training samples
- ~3,800 testing samples
- More diverse and noisy

---

## 🧠 Models Implemented

- Naive Bayes
- Logistic Regression
- Support Vector Machine (Linear SVM)

All models use **TF-IDF feature representation**.

---

## 📈 Results Summary

### BBC Dataset

| Model | Accuracy |
|--------|----------|
| Naive Bayes | 1.000 |
| Logistic Regression | 1.000 |
| SVM | 1.000 |

### AG News Dataset

| Model | Accuracy |
|--------|----------|
| Naive Bayes | 0.9739 |
| Logistic Regression | 0.9774 |
| SVM | 0.9761 |

---

## 📉 Key Insights

- BBC dataset is highly linearly separable.
- AG News dataset introduces vocabulary overlap.
- Logistic Regression slightly outperforms other models.
- Classical ML models remain highly effective for topic classification.

---

## ⚙️ How to Run

1. Clone the repository:
