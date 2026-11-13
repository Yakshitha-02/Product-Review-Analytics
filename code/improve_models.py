import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

DATA_PATH = "/home/yakshu/ProductReviewAnalytics/dataset/ml_sample_json"
MODEL_PATH = "/home/yakshu/ProductReviewAnalytics/models"
RESULT_PATH = "/home/yakshu/ProductReviewAnalytics/results"

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(RESULT_PATH, exist_ok=True)

print("Loading dataset...")

files = [
    os.path.join(DATA_PATH, f)
    for f in os.listdir(DATA_PATH)
    if f.endswith(".json")
]

df = pd.concat(
    [pd.read_json(file, lines=True) for file in files],
    ignore_index=True
)

df = df.dropna(subset=["review", "sentiment"])

X = df["review"].astype(str)
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

experiments = {
    "Naive_Bayes_a0.1": MultinomialNB(alpha=0.1),
    "Naive_Bayes_a0.5": MultinomialNB(alpha=0.5),
    "Naive_Bayes_a1.0": MultinomialNB(alpha=1.0),
    "Logistic_C0.5": LogisticRegression(C=0.5, max_iter=500),
    "Logistic_C1.0": LogisticRegression(C=1.0, max_iter=500),
    "Logistic_C2.0": LogisticRegression(C=2.0, max_iter=500),
    "Linear_SVM_C0.5": LinearSVC(C=0.5),
    "Linear_SVM_C1.0": LinearSVC(C=1.0),
    "Linear_SVM_C2.0": LinearSVC(C=2.0)
}

results = []
best_f1 = -1

for name, model in experiments.items():

    print("\nTraining:", name)

    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Macro F1:", round(f1 * 100, 2), "%")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_name = name

        joblib.dump(
            {
                "model": model,
                "vectorizer": vectorizer
            },
            os.path.join(MODEL_PATH, "best_improved_model.pkl")
        )

        print("New best model saved!")

        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1_Score", ascending=False
)

results_df.to_csv(
    os.path.join(RESULT_PATH, "improved_model_comparison.csv"),
    index=False
)

print("\n========== FINAL MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

print("\nBest Model:", best_name)
print("Best Macro F1:", round(best_f1 * 100, 2), "%")
print("\nResults saved successfully!")
