"""
A series of utility functions usefull for ML tasks.
"""

import time
from collections.abc import Iterable

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


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


def train_step_classification(
    model: nn.Module,
    trainloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    metric_fns=[],
):
    """
    Trains a PyTorch classification model for a single epoch.

    Args:
      model: A PyTorch model to train.
      data_loader: A DataLoader for the training data.
      loss_fn: A PyTorch loss function to minimize.
      optimizer: A PyTorch optimizer to use for gradient descent.
      device: A torch.device to run the model on.
      metric_fns: A list of metric functions to compute. arguments are passed based on the sklearn convention (y_true,
      y_pred.)
    Returns:
      A dictionary of the computed metrics.
    """
    model.train()
    metrics = {}
    metrics["loss"] = 0
    for metric_fn in metric_fns:
        metrics[metric_fn.__name__] = 0
    for X, y in tqdm(trainloader):
        X = X.to(device)
        y = y.to(device)
        if isinstance(X, torch.Tensor):
            y_logits = model(X)
        elif isinstance(X, Iterable):
            y_logits = model(*X)
        else:
            raise ValueError("X must be a torch.Tensor or Iterable of torch.Tensors")
        if y_logits.ndim == 1:
            y_logits = y_logits.unsqueeze(1, dim=1)
        if y_logits.ndim > 2:
            raise ValueError(
                "y_logits must be a 1D or 2D tensor but has shape "
                + str(y_logits.shape)
            )
        if y_logits.shape[1] == 1:
            y_preds = torch.round(torch.sigmoid(y_logits))
        else:
            y_preds = torch.argmax(y_logits, dim=1)
            y = torch.argmax(y, dim=1)
        loss = loss_fn(y_logits, y)
        metrics["loss"] += loss.item()
        for metric_fn in metric_fns:
            metrics[metric_fn.__name__] += metric_fn(y.cpu(), y_preds.cpu())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for metric_fn in metric_fns:
        metrics[metric_fn.__name__] /= len(trainloader)
    metrics["loss"] /= len(trainloader)
    return metrics


def test_step_classification(
    model: nn.Module,
    testloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    metric_fns=[],
):
    """
    Tests a PyTorch classification model on a dataset.

    Args:
      model: A PyTorch model to test.
      data_loader: A DataLoader for the test data.
      loss_fn: A PyTorch loss function to minimize.
      device: A torch.device to run the model on.
      metric_fns: A list of metric functions to compute. arguments are passed based on the sklearn convention (y_true,
      y_pred.)
    Returns:
      A dictionary of the computed metrics.
    """
    model.eval()
    with torch.inference_mode():
        metrics = {}
        metrics["loss"] = 0
        for metric_fn in metric_fns:
            metrics[metric_fn.__name__] = 0
        for X, y in tqdm(testloader):
            X = X.to(device)
            y = y.to(device)
            y_logits = model(X)
            y_preds = torch.argmax(y_logits, dim=1)
            loss = loss_fn(y_logits, y)
            metrics["loss"] += loss.item()
            for metric_fn in metric_fns:
                metrics[metric_fn.__name__] += metric_fn(y.cpu(), y_preds.cpu())
        for metric_fn in metric_fns:
            metrics[metric_fn.__name__] /= len(testloader)
        metrics["loss"] /= len(testloader)
    return metrics


def train_model_classification(
    model: nn.Module,
    trainloader: DataLoader,
    testloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    test_interval: int,
    epochs: int,
    metric_fns=[],
):
    """
    Trains a PyTorch classification model.

    Args:
      model: A PyTorch model to train.
      trainloader: A DataLoader for the training data.
      testloader: A DataLoader for the test data.
      loss_fn: A PyTorch loss function to minimize.
      optimizer: A PyTorch optimizer to use for gradient descent.
      device: A torch.device to run the model on.
      epochs: The number of epochs to train for.
      test_interval: The number of epochs to wait before testing the model.
      metric_fns: A list of metric functions to compute. arguments are passed based on the sklearn convention (y_true,
      y_pred.)
    Returns:
      A dictionary of the computed training and test metrics as well as the time it took to train the model.
    """
    train_metrics = {}
    test_metrics = {}
    train_metrics["loss"] = []
    test_metrics["loss"] = []
    for metric_fn in metric_fns:
        train_metrics[metric_fn.__name__] = []
        test_metrics[metric_fn.__name__] = []

    train_start = time.time()

    for epoch in tqdm(range(epochs)):
        epoch_start = time.time()
        print(f"Epoch {epoch}\n----------")
        print("Training:")
        metrics = train_step_classification(
            model, trainloader, loss_fn, optimizer, device, metric_fns
        )
        train_metrics["loss"].append(metrics["loss"])
        for metric_fn in metric_fns:
            train_metrics[metric_fn.__name__].append(metrics[metric_fn.__name__])
        for metric in train_metrics:
            print(f"    {metric}: {train_metrics[metric][-1]:.3f}")
        if epoch % test_interval == 0:
            print("Testing:")
            metrics = test_step_classification(
                model, testloader, loss_fn, device, metric_fns
            )
            test_metrics["loss"].append(metrics["loss"])
            for metric_fn in metric_fns:
                test_metrics[metric_fn.__name__].append(metrics[metric_fn.__name__])
            for metric in test_metrics:
                print(f"    {metric}: {test_metrics[metric][-1]:.3f}")
        epoch_time = time.time() - epoch_start
        print(f"Finished epoch in {epoch_time:.3f} seconds.")

    train_time = time.time() - train_start
    print(f"Finished training in {train_time:.3f} seconds.")
    return train_metrics, test_metrics, train_time
