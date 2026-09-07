import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from src.ft_linear_regression.model import model


def build_dataset_graph(
    ax: matplotlib.axes.Axes,
    normalized_mileages: np.ndarray,
    normalized_prices: np.ndarray,
):
    ax.scatter(normalized_mileages[:, 0], normalized_prices, c="g")


def build_final_model_graph(
    ax: matplotlib.axes.Axes,
    normalized_mileages: np.ndarray,
    theta: np.ndarray
):
    ax.plot(
        normalized_mileages[:, 0],
        model(normalized_mileages, theta), c="r"
    )


def build_cost_history_graph(
        ax: matplotlib.axes.Axes,
        cost_history: list
):
    ax.plot(range(len(cost_history)), cost_history)


def build_axes_titles_labels_ticks(
    axes: matplotlib.axes.Axes,
    km_min: int,
    km_max: int,
    price_min: int,
    price_max: int,
):
    xticks = np.linspace(0, 1, 5)
    xlabels = np.round(np.linspace(km_min, km_max, 5), -3).astype(int)
    axes[0].set_xticks(xticks, xlabels)

    yticks = np.linspace(0, 1, 5)
    ylabels = np.round(np.linspace(price_min, price_max, 5), -3).astype(int)
    axes[0].set_yticks(yticks, ylabels)

    axes[0].set_xlabel("mileages")
    axes[0].set_ylabel("prices")
    axes[1].set_xlabel("iterations")
    axes[1].set_ylabel("cost")

    axes[0].set_title("Linear Regression Model (Price vs Mileage)")
    axes[1].set_title("Cost Function Convergence")


def display_graph(
    normalized_mileages: np.ndarray,
    km_min: int,
    km_max: int,
    normalized_prices: np.ndarray,
    price_min: int,
    price_max: int,
    cost_history: list,
    theta: np.ndarray,
):
    _, axes = plt.subplots(2, 1)

    build_dataset_graph(axes[0], normalized_mileages, normalized_prices)
    build_final_model_graph(axes[0], normalized_mileages, theta)
    build_cost_history_graph(axes[1], cost_history)

    build_axes_titles_labels_ticks(axes, km_min, km_max, price_min, price_max)

    plt.tight_layout()
    plt.show()
