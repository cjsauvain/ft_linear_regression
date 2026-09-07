import numpy as np


def cost_function(
        mileages: np.ndarray,
        thetas: np.ndarray,
        price: np.ndarray
) -> float:
    m = len(price)
    return (1/(2 * m)) * np.sum((model(mileages, thetas) - price)**2)


def model(mileages: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return mileages.dot(theta)


def gradient(
        mileages: np.ndarray,
        prices: np.ndarray,
        theta: np.ndarray
) -> np.ndarray:
    m = len(prices)
    return (1 / m) * mileages.T.dot(model(mileages, theta) - prices)


def train_model(
        normalized_mileages: np.ndarray,
        normalized_prices: np.ndarray,
        learning_rate: float
) -> tuple[np.ndarray, list]:

    tmp_theta = np.zeros((2, 1))
    old_cost = float('inf')
    cost = cost_function(normalized_mileages, tmp_theta, normalized_prices)
    epsilon = 1e-9
    iterations = 0
    max_iter = 100000
    cost_history = []

    while abs(old_cost - cost) > epsilon and iterations < max_iter:
        old_cost = cost

        grad = gradient(normalized_mileages, normalized_prices, tmp_theta)
        tmp_theta = tmp_theta - learning_rate * grad

        cost = cost_function(normalized_mileages, tmp_theta, normalized_prices)
        cost_history.append(cost)

        iterations += 1

    return tmp_theta, cost_history
