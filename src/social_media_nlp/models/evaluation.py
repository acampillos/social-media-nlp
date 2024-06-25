from typing import Dict, Any, List

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: List[Any], y_pred: List[Any]
) -> Dict[str, float]:
    """
    Compute various classification metrics for evaluating model performance.

    Args:
        y_true (List[Any]): True labels.
        y_pred (List[Any]): Predicted labels.

    Returns:
        Dict[str, float]: Dictionary containing various classification metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision_micro = precision_score(y_true, y_pred, average="micro")
    precision_macro = precision_score(y_true, y_pred, average="macro")
    precision_weighted = precision_score(y_true, y_pred, average="weighted")
    recall_micro = recall_score(y_true, y_pred, average="micro")
    recall_macro = recall_score(y_true, y_pred, average="macro")
    recall_weighted = recall_score(y_true, y_pred, average="weighted")
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_micro": precision_micro,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_micro": recall_micro,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def generate_classification_report(y_true: List[int], y_pred: List[int]):
    """
    Generates and prints a classification report and confusion matrix based
    on true and predicted labels.

    Args:
        y_true (List[int]): True labels.
        y_pred (List[int]): Predicted labels.
    """
    report = classification_report(y_true, y_pred)
    print("Classification Report:")
    print(report)

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
