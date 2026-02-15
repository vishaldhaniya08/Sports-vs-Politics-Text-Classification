"""
Problem 4: Sports vs Politics Classification
Using TWO datasets:
1. BBC Dataset (clean dataset)
2. AG News Dataset (filtered to World & Sports)

This script:
- Loads both datasets
- Preprocesses and splits data
- Extracts TF-IDF features
- Trains 3 models (NB, LR, SVM)
- Prints accuracy and classification reports
- Compares performance across datasets

Author: Vishal
Roll No: B23CM1048
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# -------------------------------------------------------
# 1. LOAD BBC DATASET
# -------------------------------------------------------

def load_bbc_dataset(base_path):
    texts = []
    labels = []

    for label in ["sport", "politics"]:
        folder_path = os.path.join(base_path, label)

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r', encoding='latin-1') as f:
                texts.append(f.read())
                labels.append(label)

    return texts, labels


# -------------------------------------------------------
# 2. LOAD AG NEWS DATASET (FILTER CLASS 1 & 2)
# -------------------------------------------------------

def load_ag_dataset(base_path):

    train_df = pd.read_csv(os.path.join(base_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(base_path, "test.csv"))

    # AG News class mapping:
    # 1 = World (politics-like)
    # 2 = Sports
    # 3 = Business
    # 4 = Sci/Tech

    train_df = train_df[train_df["Class Index"].isin([1, 2])]
    test_df = test_df[test_df["Class Index"].isin([1, 2])]

    # Relabel classes
    train_df["label"] = train_df["Class Index"].map({1: "politics", 2: "sport"})
    test_df["label"] = test_df["Class Index"].map({1: "politics", 2: "sport"})

    # Combine title + description
    train_texts = train_df["Title"] + " " + train_df["Description"]
    test_texts = test_df["Title"] + " " + test_df["Description"]

    return list(train_texts), list(test_texts), list(train_df["label"]), list(test_df["label"])


# -------------------------------------------------------
# 3. TRAIN & EVALUATE FUNCTION
# -------------------------------------------------------

def train_and_evaluate(X_train, X_test, y_train, y_test, dataset_name):

    print("\n===============================")
    print(f"DATASET: {dataset_name}")
    print("===============================")

    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    results = []

    # ---------------- Naive Bayes ----------------
    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    y_pred_nb = nb.predict(X_test_vec)

    acc_nb = accuracy_score(y_test, y_pred_nb)
    print("\nNaive Bayes Accuracy:", acc_nb)
    print(classification_report(y_test, y_pred_nb))

    results.append(("Naive Bayes", acc_nb))

    # ---------------- Logistic Regression ----------------
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_vec, y_train)
    y_pred_lr = lr.predict(X_test_vec)

    acc_lr = accuracy_score(y_test, y_pred_lr)
    print("\nLogistic Regression Accuracy:", acc_lr)
    print(classification_report(y_test, y_pred_lr))

    results.append(("Logistic Regression", acc_lr))

    # ---------------- SVM ----------------
    svm = LinearSVC()
    svm.fit(X_train_vec, y_train)
    y_pred_svm = svm.predict(X_test_vec)

    acc_svm = accuracy_score(y_test, y_pred_svm)
    print("\nSVM Accuracy:", acc_svm)
    print(classification_report(y_test, y_pred_svm))

    results.append(("SVM", acc_svm))

    print("\n---- Summary for", dataset_name, "----")
    for model, acc in results:
        print(f"{model:<25} Accuracy: {acc:.4f}")

    return results


# -------------------------------------------------------
# 4. MAIN FUNCTION
# -------------------------------------------------------

def main():

    # ---------------- BBC Dataset ----------------
    bbc_texts, bbc_labels = load_bbc_dataset("bbc")

    X_train_bbc, X_test_bbc, y_train_bbc, y_test_bbc = train_test_split(
        bbc_texts,
        bbc_labels,
        test_size=0.2,
        random_state=42,
        stratify=bbc_labels
    )

    bbc_results = train_and_evaluate(
        X_train_bbc, X_test_bbc,
        y_train_bbc, y_test_bbc,
        "BBC Dataset"
    )

    # ---------------- AG Dataset ----------------
    ag_train_texts, ag_test_texts, ag_train_labels, ag_test_labels = load_ag_dataset("ag")

    ag_results = train_and_evaluate(
        ag_train_texts, ag_test_texts,
        ag_train_labels, ag_test_labels,
        "AG News Dataset"
    )

    # ---------------------------------------------------
    # FINAL COMPARISON TABLE
    # ---------------------------------------------------

    print("\n================ FINAL COMPARISON ================")
    print("{:<25} {:<15} {:<10}".format("Model", "Dataset", "Accuracy"))
    print("---------------------------------------------------")

    for model, acc in bbc_results:
        print("{:<25} {:<15} {:.4f}".format(model, "BBC", acc))

    for model, acc in ag_results:
        print("{:<25} {:<15} {:.4f}".format(model, "AG News", acc))


if __name__ == "__main__":
    main()
