
import os
import re
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FAKE_PATH = os.path.join(BASE_DIR, "dataset", "Fake.csv")
TRUE_PATH = os.path.join(BASE_DIR, "dataset", "True.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Labels
FAKE_LABEL = 0
REAL_LABEL = 1


# Clean news text
def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Load dataset
def load_data():

    if not os.path.exists(FAKE_PATH):
        raise FileNotFoundError(f"Missing file: {FAKE_PATH}")

    if not os.path.exists(TRUE_PATH):
        raise FileNotFoundError(f"Missing file: {TRUE_PATH}")

    fake_df = pd.read_csv(FAKE_PATH)
    true_df = pd.read_csv(TRUE_PATH)

    fake_df["label"] = FAKE_LABEL
    true_df["label"] = REAL_LABEL

    return pd.concat(
        [fake_df, true_df],
        ignore_index=True
    )


# Prepare dataset
def prepare_data():

    df = load_data()

    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    df["content"] = (
        df["title"] + " " + df["text"]
    ).apply(clean_text)

    df = df[["content", "label"]]

    df = df[df["content"].str.len() > 0]

    df = df.drop_duplicates(subset="content")

    return df.reset_index(drop=True)


# Train machine learning model
def train_model():

    print("\nLoading dataset...")

    data = prepare_data()

    X = data["content"]
    y = data["label"]

    print("Total articles:", len(data))

    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nApplying TF-IDF...")

    vectorizer = TfidfVectorizer(
        max_features=50000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train_tfidf, y_train)

    print("\nTesting model...")

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)

    print("\n========== MODEL RESULTS ==========")

    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")

    print(classification_report(
        y_test,
        predictions,
        target_names=["FAKE", "REAL"]
    ))

    print("\nConfusion Matrix:")

    print(confusion_matrix(y_test, predictions))

    # Save trained model
    joblib.dump(
        model,
        os.path.join(MODEL_DIR, "fake_news_model.pkl")
    )

    joblib.dump(
        vectorizer,
        os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    )

    print("\nModel trained successfully!")

    print("Model saved inside models folder.")


if __name__ == "__main__":
    train_model()
