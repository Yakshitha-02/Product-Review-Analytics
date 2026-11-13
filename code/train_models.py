
import pandas as pd
import glob
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

print("Loading dataset...")

file_path = glob.glob(
    "/home/yakshu/ProductReviewAnalytics/dataset/ml_sample_json/*.json"
)[0]

df = pd.read_json(file_path, lines=True)

X = df["review"].fillna("")
y = df["sentiment"]

print("Total records:", len(df))
print("\nClass distribution:")
print(y.value_counts())

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nConverting text into TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=3000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF feature shape:", X_train_tfidf.shape)

models = {
    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=300,
        solver="lbfgs"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=50,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=2
    )
}

results = []

os.makedirs(
    "/home/yakshu/ProductReviewAnalytics/results",
    exist_ok=True
)

os.makedirs(
    "/home/yakshu/ProductReviewAnalytics/models",
    exist_ok=True
)

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=["Negative", "Neutral", "Positive"]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Neutral", "Positive"]
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title(name + " Confusion Matrix")
    plt.tight_layout()

    safe_name = name.lower().replace(" ", "_")

    plt.savefig(
        f"/home/yakshu/ProductReviewAnalytics/results/{safe_name}_confusion_matrix.png"
    )

    plt.close()

    joblib.dump(
        {"model": model, "vectorizer": vectorizer},
        f"/home/yakshu/ProductReviewAnalytics/models/{safe_name}.pkl"
    )

    print(name, "completed!")

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

results_df.to_csv(
    "/home/yakshu/ProductReviewAnalytics/results/model_comparison.csv",
    index=False
)

best_model = results_df.iloc[0]

print("\n========== BEST MODEL ==========")
print("Model:", best_model["Model"])
print("Accuracy:", best_model["Accuracy"])
print("Precision:", best_model["Precision"])
print("Recall:", best_model["Recall"])
print("F1 Score:", best_model["F1 Score"])

print("\nAll model results and trained models saved successfully!")
