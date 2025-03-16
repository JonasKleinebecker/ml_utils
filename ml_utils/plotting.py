"""
A series of utility functions usefull for ML tasks.
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


def plot_decision_boundary(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> None:
    """Plots the decision boundary of a PyTorch model for a 2D feature space.

    This function visualizes the decision boundary of a trained PyTorch model by creating a grid of points
    across the feature space, making predictions on these points, and plotting the results. It is particularly
    useful for binary and multi-class classification tasks with 2D input features.

    Args:
        model (nn.Module): A trained PyTorch model that takes 2D input features and outputs logits.
                           The model should be capable of making predictions on the provided data.
        X (torch.Tensor): A 2D tensor of shape (n_samples, 2) containing the input features.
                           This represents the feature space to visualize.
        y (torch.Tensor): A 1D tensor of shape (n_samples,) containing the true labels for the input features.
                           These labels are used to color the scatter plot of the data points.

    Returns:
        None: This function does not return any value. It directly displays a matplotlib plot.

    Example:
        >>> # Assuming `model` is a trained PyTorch model and `X`, `y` are your dataset
        >>> X = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])  # Example 2D input features
        >>> y = torch.tensor([0, 1, 0])  # Example binary labels
        >>> plot_decision_boundary(model, X, y)
        >>> # This will display a plot with the decision boundary and data points colored by their true labels.

    Notes:
        - The function assumes the input features `X` are 2D (i.e., each sample has exactly 2 features).
        - The model's predictions are converted to class labels using `torch.softmax` for multi-class classification
          or `torch.sigmoid` for binary classification.
        - The plot uses `plt.contourf` to fill the decision regions and `plt.scatter` to overlay the data points.
        - The function moves the model and data to the CPU for compatibility with matplotlib."""
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
