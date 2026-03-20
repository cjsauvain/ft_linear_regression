import sys
import pandas as pd


def validate_args() -> tuple[str, int]:
    args = sys.argv

    assert len(args) == 3, f"Usage: python learn.py <dataset_filepath> <learning_rate>"
    assert args[2].isdigit(), "learning rate must be an integer"

    return args[1], int(args[2])


def build_numpy_array_from_dataset(dataset_filepath: str) -> pd.DataFrame:
    dataset_df = pd.read_csv(dataset_filepath)

    return dataset_df


# def get_latest_theta0_theta1() -> tuple[float, float]:
#     if not os.path.exists("thetas.txt"):
#         return 0., 0.

#     with open("thetas.txt", "r") as file:
#         content = file.read().strip()

#         if not re.fullmatch(r"^theta0=-?\d+(\.\d+)?, theta1=-?\d+(\.\d+)?$", content):
#             print("thetas.txt should be formatted as followed: theta0=<float>, theta1=<float>")
#             sys.exit(1)
        
#         theta0 = float(content.split(", ")[0].split("=")[1])
#         theta1 = float(content.split(", ")[1].split("=")[1])

#     return theta0, theta1


def process_linear_regression(dataset_array: pd.DataFrame, learning_rate: int) -> list[float]:
    mileages = dataset_array["km"].to_numpy()
    prices = dataset_array["price"].to_numpy()

    theta0 = 0
    theta1 = 0
    sum = 0
    for i in range(learning_rate):
        sum += ()

    return [0., 0.]


def train():
    try:
        dataset_filepath, learning_rate = validate_args()
        dataset_array = build_numpy_array_from_dataset(dataset_filepath)
        # theta0, theta1 = get_latest_theta0_theta1()
        theta0, theta1 = process_linear_regression(dataset_array, learning_rate)

        print(theta0, theta1)
    
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    train()