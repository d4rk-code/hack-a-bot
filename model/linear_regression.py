# model/linear_regression.py
"""
Linear regression from scratch (gradient descent) with dynamic learning rate.
Usage:
    from model.linear_regression import train_linear_regression, predict

    m, b, loss_history = train_linear_regression(
        path="data/linear_regression_dataset.csv",
        x_col="Study_Hours",
        y_col="Score",
        learning_rate=0.0001,
        epochs=2000,
        lr_decay={'every':100, 'factor':0.9}
    )
    preds = predict([1,2,3], m, b)
"""

import pandas as pd
from typing import List, Tuple, Dict

def train_and_predict(path: str,
                            x_col: str,
                            y_col: str,
                            learning_rate: float = 0.0001,
                            epochs: int = 1000,
                            lr_decay: Dict[str, float] = None
                           ) -> Tuple[float, float, List[float]]:
    """
    Trains linear regression y = m*x + b using batch gradient descent.
    Args:
        path: path to CSV dataset
        x_col, y_col: column names to use
        learning_rate: initial learning rate
        epochs: number of iterations
        lr_decay: optional dict controlling decay:
            {'every': 100, 'factor': 0.9} -> multiply lr by 0.9 every 100 epochs
    Returns:
        m, b, loss_history (list of MSE values)
    """
    df = pd.read_csv(path)
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError("x_col or y_col not in dataset")

    X = [float(v) for v in df[x_col].to_list()]
    Y = [float(v) for v in df[y_col].to_list()]

    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset")

    m = 0.0
    b = 0.0
    L = float(learning_rate)
    loss_history: List[float] = []

    # default decay if not provided: reduce 10% every 100 epochs
    if lr_decay is None:
        lr_decay = {'every': 100, 'factor': 0.9}

    for epoch in range(1, epochs + 1):
        grad_m = 0.0
        grad_b = 0.0
        for xi, yi in zip(X, Y):
            error = yi - (m * xi + b)
            grad_m += -(2 / n) * xi * error
            grad_b += -(2 / n) * error

        m -= L * grad_m
        b -= L * grad_b

        # compute loss (MSE)
        mse = sum((yi - (m * xi + b))**2 for xi, yi in zip(X, Y)) / n
        loss_history.append(mse)

        # dynamic learning rate decay schedule
        every = lr_decay.get('every', None)
        factor = lr_decay.get('factor', 1.0)
        if every and epoch % every == 0 and epoch > 0:
            L *= factor

    return m, b, loss_history

def predict(X: List[float], m: float, b: float) -> List[float]:
    """Predict Y for list X using parameters m and b."""
    return [m * float(x) + b for x in X]

