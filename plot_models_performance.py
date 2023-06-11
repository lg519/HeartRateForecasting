import numpy as np
import matplotlib.pyplot as plt

# Assuming these are your models and their performance metrics
models = ["Baseline", "RNN", "GRU", "LSTM"]

x = np.arange(len(models))  # set x-axis with number of models
width = 0.3
metric_name = "mean_absolute_error"

# Manually adjust the values
val_performance = {
    "Baseline": 0.0569,
    "RNN": 0.0460,
    "GRU": 0.0311,
    "LSTM": 0.0334,
}  # replace random values with validation performance
performance = {
    "Baseline": 0.0546,
    "RNN": 0.0346,
    "GRU": 0.0318,
    "LSTM": 0.0368,
}  # replace random values with test performance


val_mae = [val_performance[model] for model in models]
test_mae = [performance[model] for model in models]

plt.ylabel("Mean Absolute Error (MAE)")
plt.bar(x - 0.17, val_mae, width, label="Validation")
plt.bar(x + 0.17, test_mae, width, label="Test")
plt.xticks(ticks=x, labels=models, rotation=45)
_ = plt.legend()
plt.show()
