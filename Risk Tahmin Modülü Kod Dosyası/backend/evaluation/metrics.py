from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import numpy as np


def evaluate_model(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
        "confusion": confusion_matrix(y_true, y_pred).tolist()
    }

    try:
        metrics["roc_auc"] = roc_auc_score(
            y_true,
            np.eye(len(set(y_true)))[y_pred],
            multi_class="ovr"
        )
    except:
        metrics["roc_auc"] = None

    return metrics
