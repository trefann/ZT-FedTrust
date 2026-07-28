"""Plotting utilities for visualizing federated training progress."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; must be set before pyplot import
import matplotlib.pyplot as plt


def plot_accuracy(history: List[Dict[str, float]], output_path: str) -> None:
    """Plot global test accuracy per round and save it as a PNG."""
    rounds = [entry["round"] for entry in history]
    accuracy = [entry["test_accuracy"] for entry in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, accuracy, marker="o", color="tab:blue")
    ax.set_xlabel("Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Global Model Accuracy per Round")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_loss(history: List[Dict[str, float]], output_path: str) -> None:
    """Plot average client train loss and global test loss per round, save as a PNG."""
    rounds = [entry["round"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    test_loss = [entry["test_loss"] for entry in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, train_loss, marker="o", color="tab:orange", label="Train Loss")
    ax.plot(rounds, test_loss, marker="o", color="tab:red", label="Test Loss")
    ax.set_xlabel("Round")
    ax.set_ylabel("Loss")
    ax.set_title("Loss per Round")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_plots(history: List[Dict[str, float]], results_dir: str) -> None:
    """Generate and save the accuracy and loss plots for a completed simulation."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    plot_accuracy(history, str(Path(results_dir) / "accuracy_plot.png"))
    plot_loss(history, str(Path(results_dir) / "loss_plot.png"))
