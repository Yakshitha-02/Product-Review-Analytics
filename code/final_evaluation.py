
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

DATA_PATH = "/home/yakshu/ProductReviewAnalytics/dataset/ml_sample_json"
RESULT_PATH = "/home/yakshu/ProductReviewAnalytics/results"
MODEL_PATH = "/home/yakshu/ProductReviewAnalytics/models"

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

print("Total records:", len(df))

# 70% training and 30% temporary data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Split temporary data equally into validation and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))

print("\nCreating combined TF-IDF features...")

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
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

print("\nTraining candidate models...")

candidate_values = [0.1, 0.25, 0.5, 0.75, 1.0]

validation_results = []
best_f1 = -1
best_c = None

for c in candidate_values:

    model = LogisticRegression(C=c, max_iter=500)
    model.fit(X_train_vec, y_train)

    val_predictions = model.predict(X_val_vec)

    val_f1 = f1_score(
        y_val, val_predictions,
        average="macro",
        zero_division=0
    )

    val_accuracy = accuracy_score(y_val, val_predictions)

    validation_results.append({
        "C": c,
        "Validation_Accuracy": val_accuracy,
        "Validation_Macro_F1": val_f1
    })

    print(
        "C:", c,
        "| Validation Accuracy:", round(val_accuracy * 100, 2),
        "| Validation Macro F1:", round(val_f1 * 100, 2)
    )

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_c = c

print("\nSelected C:", best_c)

# Refit using training + validation data
X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])

final_vectorizer = FeatureUnion([
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

X_train_full_vec = final_vectorizer.fit_transform(X_train_full)
X_test_final_vec = final_vectorizer.transform(X_test)

final_model = LogisticRegression(C=best_c, max_iter=500)
final_model.fit(X_train_full_vec, y_train_full)

print("\nEvaluating on untouched test set...")

test_predictions = final_model.predict(X_test_final_vec)

accuracy = accuracy_score(y_test, test_predictions)
precision = precision_score(
    y_test, test_predictions,
    average="macro",
    zero_division=0
)
recall = recall_score(
    y_test, test_predictions,
    average="macro",
    zero_division=0
)
f1 = f1_score(
    y_test, test_predictions,
    average="macro",
    zero_division=0
)

print("\n========== FINAL TEST RESULTS ==========")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Macro Precision:", round(precision * 100, 2), "%")
print("Macro Recall:", round(recall * 100, 2), "%")
print("Macro F1:", round(f1 * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, test_predictions))

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=["Negative", "Neutral", "Positive"]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Neutral", "Positive"]
)

display.plot(cmap="Blues", values_format="d")
plt.title("Final Sentiment Classification Confusion Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULT_PATH, "final_confusion_matrix.png"),
    dpi=300
)

plt.close()

pd.DataFrame(validation_results).to_csv(
    os.path.join(RESULT_PATH, "final_validation_results.csv"),
    index=False
)

final_metrics = pd.DataFrame([{
    "Selected_C": best_c,
    "Accuracy": accuracy,
    "Macro_Precision": precision,
    "Macro_Recall": recall,
    "Macro_F1": f1
}])

final_metrics.to_csv(
    os.path.join(RESULT_PATH, "final_test_metrics.csv"),
    index=False
)

joblib.dump(
    {
        "model": final_model,
        "vectorizer": final_vectorizer,
        "selected_C": best_c
    },
    os.path.join(MODEL_PATH, "final_sentiment_model.pkl")
)

print("\nFinal model, metrics and confusion matrix saved!")
