import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

DATA_PATH = "/home/yakshu/ProductReviewAnalytics/dataset/ml_sample_json"
MODEL_PATH = "/home/yakshu/ProductReviewAnalytics/models"
RESULT_PATH = "/home/yakshu/ProductReviewAnalytics/results"

files = [
    os.path.join(DATA_PATH, f)
    for f in os.listdir(DATA_PATH)
    if f.endswith(".json")
]

df = pd.concat(
    [pd.read_json(file, lines=True) for file in files],
    ignore_index=True
)

X = df["review"].astype(str)
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Creating combined TF-IDF features...")

vectorizer = FeatureUnion([
    ("word", TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )),
    ("char", TfidfVectorizer(
        analyzer="char",
        max_features=10000,
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True
    ))
])

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Training Logistic Regression...")

model = LogisticRegression(C=0.5, max_iter=500)
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

print("\nCOMBINED TF-IDF RESULTS")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Macro Precision:", round(precision * 100, 2), "%")
print("Macro Recall:", round(recall * 100, 2), "%")
print("Macro F1:", round(f1 * 100, 2), "%")

joblib.dump(
    {"model": model, "vectorizer": vectorizer},
    os.path.join(MODEL_PATH, "combined_tfidf_model.pkl")
)

pd.DataFrame([{
    "Experiment": "Combined_Word_Character_TFIDF",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1
}]).to_csv(
    os.path.join(RESULT_PATH, "combined_tfidf_results.csv"),
    index=False
)

print("\nModel and results saved successfully!")
