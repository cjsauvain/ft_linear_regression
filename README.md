# ft_linear_regression

Predict a car price from its mileage. The model is a linear regression trained with gradient descent.

Mileage and price are scaled to `[0, 1]`. The prediction is `price = θ0 + θ1 * mileage`, then scaled back to euros. Training stops when the cost barely changes, and saves `data/thetas.json` and `data/min_max.json`.

Run every command from the repository root.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python train.py data/data.csv 0.1
python estimate.py
python model_performance.py data/data.csv
```

`train.py` opens two plots: the fitted line, and the cost over iterations. `estimate.py` asks for a mileage and prints a price. `model_performance.py` prints the R² score.
