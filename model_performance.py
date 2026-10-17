import sys

import numpy as np

from src.ft_linear_regression.data import (
    normalize_data,
    retrieve_data_from_dataset,
    retrieve_thetas,
)
from src.ft_linear_regression.model import model


def validate_args() -> str:
    args = sys.argv

    if len(args) != 2:
        raise ValueError("Usage: python model_performance.py <dataset_filepath>")

    return args[1]


def r2_score(prices: np.ndarray, final_model: np.ndarray):
    u = ((prices - final_model) ** 2).sum()
    v = ((prices - prices.mean()) ** 2).sum()

    print(f"Model Performance (R2-score): {(1 - u / v):.2f}")


def process_performance():
    try:
        dataset_filepath = validate_args()

        mileages, km_min, km_max, prices, price_min, price_max = (
            retrieve_data_from_dataset(dataset_filepath)
        )

        normalized_mileages, normalized_prices = normalize_data(
            mileages, km_min, km_max, prices, price_min, price_max
        )
        normalized_mileages = np.hstack(
            (normalized_mileages, np.ones(normalized_mileages.shape))
        )

        theta0, theta1 = retrieve_thetas()
        thetas = np.array([theta1, theta0])
        thetas = thetas.reshape(thetas.shape[0], -1)

        r2_score(normalized_prices, model(normalized_mileages, thetas))

    except Exception as e:
        print(f"{type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    process_performance()
