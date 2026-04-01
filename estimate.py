from src.ft_linear_regression.data import retrieve_thetas, retrieve_min_max


def estimate():
    try:
        user_input = input("Enter a mileage: ")
        assert user_input.isdigit(), "You must provide a positive integer"
        user_input = int(user_input)

        theta0, theta1 = retrieve_thetas()
        estimatedPrice = 0
        if theta0 != 0 and theta1 != 0:
            km_min, km_max, price_min, price_max = retrieve_min_max()
            normalized_km = (user_input - km_min) / (km_max - km_min)
            normalized_price = theta0 + theta1 * normalized_km
            estimatedPrice = \
                normalized_price * (price_max - price_min) + price_min
        print(estimatedPrice)

    except Exception as e:
        print(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    estimate()
