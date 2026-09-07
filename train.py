import sys

import numpy as np

from src.ft_linear_regression.data import (
    normalize_data,
    retrieve_data_from_dataset,
    write_min_max_to_json_file,
    write_thetas_to_json_file,
)
from src.ft_linear_regression.model import train_model
from src.ft_linear_regression.visualization import display_graph


def validate_args() -> tuple[str, float]:
    args = sys.argv

    if len(args) != 3:
        raise ValueError(
            "Usage: python learn.py <dataset_filepath> <learning_rate>"
        )

    return args[1], float(args[2])


def process_linear_regression(
        normalized_mileages: np.ndarray, km_min: int, km_max: int,
        normalized_prices: np.ndarray, price_min: int, price_max: int,
        learning_rate: float
) -> tuple[float, float]:

    normalized_mileages = np.hstack(
        (normalized_mileages, np.ones(normalized_mileages.shape))
    )

    theta, cost_history = train_model(
        normalized_mileages, normalized_prices,
        learning_rate
    )

    display_graph(
        normalized_mileages, km_min, km_max,
        normalized_prices, price_min, price_max,
        cost_history, theta
    )

    return theta[1][0], theta[0][0]


def train():
    try:
        dataset_filepath, learning_rate = validate_args()

        mileages, km_min, km_max, prices, price_min, price_max \
            = retrieve_data_from_dataset(dataset_filepath)

        normalized_mileages, normalized_prices = normalize_data(
            mileages, km_min, km_max,
            prices, price_min, price_max
        )

        theta0, theta1 = process_linear_regression(
            normalized_mileages, km_min, km_max,
            normalized_prices, price_min, price_max,
            learning_rate
        )

        write_thetas_to_json_file(theta0, theta1)
        write_min_max_to_json_file(km_min, km_max, price_min, price_max)

    except Exception as e:
        print(f"{type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    train()
