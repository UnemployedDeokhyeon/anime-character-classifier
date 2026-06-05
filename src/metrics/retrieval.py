import numpy as np


def top_k_accuracy(retrieved_labels: np.ndarray, query_labels: np.ndarray, k: int = 1) -> float:
    """Fraction of queries where correct label appears in top-k results."""
    correct = 0
    for q_label, ret_labels in zip(query_labels, retrieved_labels):
        if q_label in ret_labels[:k]:
            correct += 1
    return correct / len(query_labels)


def mean_average_precision(retrieved_labels: np.ndarray, query_labels: np.ndarray) -> float:
    """Mean Average Precision for retrieval evaluation."""
    aps = []
    for q_label, ret_labels in zip(query_labels, retrieved_labels):
        hits, precision_sum = 0, 0.0
        for rank, label in enumerate(ret_labels, 1):
            if label == q_label:
                hits += 1
                precision_sum += hits / rank
        aps.append(precision_sum / max(hits, 1))
    return float(np.mean(aps))
