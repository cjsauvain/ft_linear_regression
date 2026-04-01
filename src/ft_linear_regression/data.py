import json
import numpy as np
import pandas as pd


def retrieve_thetas() -> tuple[float, float]:
    try:
        with open("thetas.json", "r") as file:
            data = json.load(file)

            return float(data["theta0"]), float(data["theta1"])

    except Exception as e:
        if type(e) is FileNotFoundError:
            return 0., 0.
        raise e


def retrieve_min_max() -> tuple[int, int, int, int]:
    with open("min_max.json", "r") as file:
        data = json.load(file)

        if type(data["km_min"]) is not int \
                or type(data["km_max"]) is not int:
            raise TypeError("values should be integers, retrain your model")
        if type(data["price_min"]) is not int \
                or type(data["price_max"]) is not int:
            raise TypeError("values should be integers, retrain your model")

        return (
            data["km_min"], data["km_max"],
            data["price_min"], data["price_max"]
        )


def write_thetas_to_json_file(theta0: float, theta1: float):
    json_content: dict[str, float] = {"theta0": theta0, "theta1": theta1}

    with open("thetas.json", "w") as file:
        json.dump(json_content, file, indent=4)


def write_min_max_to_json_file(
        km_min: int, km_max: int,
        price_min: int, price_max: int
):
    json_content: dict[str, int] = {
        "km_min": km_min,
        "km_max": km_max,
        "price_min": price_min,
        "price_max": price_max
    }

    with open("min_max.json", "w") as file:
        json.dump(json_content, file, indent=4)


def retrieve_data_from_dataset(
        dataset_filepath: str
) -> tuple[np.ndarray, int, int, np.ndarray, int, int]:
    dataset_df = pd.read_csv(dataset_filepath)

    mileages = dataset_df["km"]
    mileages = np.array(mileages)
    mileages = np.reshape(mileages, (mileages.shape[0], 1))

    km_min = int(np.min(mileages))
    km_max = int(np.max(mileages))
    if km_max == km_min:
        raise TypeError("All mileages are identical")

    prices = np.array(dataset_df["price"])
    prices = np.reshape(prices, (prices.shape[0], 1))

    price_min = int(np.min(prices))
    price_max = int(np.max(prices))
    if price_max == price_min:
        raise TypeError("All prices are identical")

    return mileages, km_min, km_max, prices, price_min, price_max


def normalize_data(
        mileages: np.ndarray, km_min: int, km_max: int,
        prices: np.ndarray, price_min: int, price_max: int
) -> tuple[np.ndarray, np.ndarray]:

    normalized_mileages = (mileages - km_min) / (km_max - km_min)
    normalized_prices = (prices - price_min) / (price_max - price_min)

    return normalized_mileages, normalized_prices
