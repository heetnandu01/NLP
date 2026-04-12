"""Main script for the Opinion Mining System."""

from __future__ import annotations

import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from utils import preprocess_corpus, preprocess_text

warnings.filterwarnings("ignore", category=UserWarning)

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")
CONFUSION_MATRIX_PATH = os.path.join(os.path.dirname(__file__), "confusion_matrix.png")


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the dataset from CSV and validate the required columns."""
    data = pd.read_csv(file_path)
    required_columns = {"text", "label"}

    if not required_columns.issubset(data.columns):
        raise ValueError("dataset.csv must contain 'text' and 'label' columns.")

    return data.dropna(subset=["text", "label"]).copy()


def train_model(data: pd.DataFrame):
    """Preprocess data, split it, and train the sentiment classifier."""
    data["processed_text"] = preprocess_corpus(data["text"])

    features = data["processed_text"]
    labels = data["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer()
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train_tfidf, y_train)

    y_pred = model.predict(x_test_tfidf)

    return {
        "vectorizer": vectorizer,
        "model": model,
        "x_test": x_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def evaluate_model(y_test, y_pred) -> None:
    """Print accuracy and the classification report."""
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))


def plot_confusion_matrix(y_test, y_pred) -> None:
    """Plot and save a confusion matrix using matplotlib."""
    labels = ["Positive", "Negative", "Neutral"]
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(7, 5))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_positions = range(len(labels))
    plt.xticks(tick_positions, labels, rotation=45)
    plt.yticks(tick_positions, labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            plt.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
                color="black",
            )

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300, bbox_inches="tight")
    plt.show()


def predict_feedback(model, vectorizer, feedback: str) -> str:
    """Predict the sentiment label for a single piece of feedback."""
    processed_feedback = preprocess_text(feedback)
    transformed_feedback = vectorizer.transform([processed_feedback])
    prediction = model.predict(transformed_feedback)[0]
    return prediction


def print_cli_header() -> None:
    """Display a clean CLI header."""
    print("\n" + "=" * 64)
    print(" Opinion Mining System - Smart Governance Feedback Analyzer")
    print("=" * 64)


def get_feedback_by_index(data: pd.DataFrame, user_input: str) -> str | None:
    """Return dataset feedback text by 1-based sentence index."""
    cleaned_input = user_input.strip()
    if not cleaned_input:
        print("Please enter a sentence number.")
        return None

    if not cleaned_input.isdigit():
        print("Invalid input. Please enter a numeric sentence number.")
        return None

    sentence_number = int(cleaned_input)
    max_sentence_number = len(data)
    if sentence_number < 1 or sentence_number > max_sentence_number:
        print(
            f"Number exceeded or not found. Please enter a number between 1 and {max_sentence_number}."
        )
        return None

    selected_text = str(data.iloc[sentence_number - 1]["text"])
    print(f"Selected sentence #{sentence_number}: {selected_text}")
    return selected_text


def show_dataset_preview(data: pd.DataFrame, preview_size: int = 5) -> None:
    """Show a small, readable preview of dataset sentences with index numbers."""
    print("\nDataset Preview (first few rows):")
    print("-" * 64)

    preview = data.head(preview_size).copy()
    for idx, row in preview.iterrows():
        sentence_number = idx + 1
        print(f"{sentence_number:>2}. [{row['label']}] {row['text']}")

    print("-" * 64)
    print(f"Total dataset sentences: {len(data)}")


def run_cli(model, vectorizer, data: pd.DataFrame) -> None:
    """Provide a clean terminal menu for real-time predictions."""
    total_sentences = len(data)

    while True:
        print_cli_header()
        print("Choose an option:")
        print("1. Analyze custom feedback text")
        print("2. Analyze by dataset sentence number")
        print("3. View dataset preview")
        print("4. Exit")

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            feedback = input("Enter feedback text: ").strip()
            if not feedback:
                print("Please enter some feedback text.")
                continue

            prediction = predict_feedback(model, vectorizer, feedback)
            print(f"Predicted Sentiment: {prediction}")

        elif choice == "2":
            print(f"Enter a sentence number from 1 to {total_sentences}.")
            sentence_input = input("Sentence number: ")
            feedback = get_feedback_by_index(data, sentence_input)
            if feedback is None:
                continue

            prediction = predict_feedback(model, vectorizer, feedback)
            print(f"Predicted Sentiment: {prediction}")

        elif choice == "3":
            show_dataset_preview(data)

        elif choice == "4":
            print("Exiting the program.")
            break

        else:
            print("Invalid choice. Please select 1, 2, 3, or 4.")


def main() -> None:
    """Run the complete opinion mining workflow."""
    data = load_dataset(DATASET_PATH)
    results = train_model(data)

    evaluate_model(results["y_test"], results["y_pred"])
    plot_confusion_matrix(results["y_test"], results["y_pred"])
    run_cli(results["model"], results["vectorizer"], data)


if __name__ == "__main__":
    main()
