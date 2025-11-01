# model/knn_model.py
"""
Simple K-Nearest Neighbours predictor (functional).
Usage:
    from model.knn_model import knn_predict
    pred, neighbors = knn_predict(
        point=[25000, 6.5],
        path="data/car_knn_dataset.csv",
        x_col="price_kUSD",
        y_col="common",
        label_col="brand",
        k=4
    )
"""

import pandas as pd
import math
from typing import List, Tuple, Any

def knn_predict(point: List[float],
                path: str,
                x_col: str,
                y_col: str,
                label_col: str,
                k: int = 4) -> Tuple[Any, List[Tuple[float, Any, float, float]]]:
    """
    Predict label for `point` = [x_val, y_val] using simple Euclidean KNN.

    Returns:
        predicted_label, nearest_neighbors
        nearest_neighbors: list of tuples (distance, label, x_val, y_val)
    """
    df = pd.read_csv(path)
    if x_col not in df.columns or y_col not in df.columns or label_col not in df.columns:
        raise ValueError("One or more specified columns not found in dataset.")

    x_vals = df[x_col].to_list()
    y_vals = df[y_col].to_list()
    labels = df[label_col].to_list()

    distances = []
    for xi, yi, lab in zip(x_vals, y_vals, labels):
        try:
            dist = math.hypot(xi - point[0], yi - point[1])
        except Exception:
            # ensure numeric
            dist = ((float(xi) - float(point[0]))**2 + (float(yi) - float(point[1]))**2)**0.5
        distances.append((dist, lab, xi, yi))

    distances.sort(key=lambda t: t[0])
    nearest = distances[:k]
    neighbor_labels = [lab for _, lab, _, _ in nearest]

    # majority vote (ties resolved by first max)
    predicted = max(set(neighbor_labels), key=neighbor_labels.count)
    return predicted, nearest

