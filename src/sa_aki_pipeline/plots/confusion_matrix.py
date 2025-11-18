"""Matplotlib helpers for plotting confusion matrices."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    matrix: np.ndarray,
    classes: list[str],
    title: str = "Confusion Matrix",
    cmap: str = "YlGnBu",
):
    """Render a labeled confusion matrix."""

    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    im = ax.imshow(matrix, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > thresh else "black",
            )

    fig.tight_layout()
    return ax
