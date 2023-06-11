import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.helpers.window_generator import (
    WindowGenerator,
    num_output_features,
    input_width,
    label_width,
    shift,
)


# load the model
model = tf.keras.models.load_model("saved_models/LSTM_hyperparameter_best")


# Create a window
window = WindowGenerator(input_width=input_width, label_width=label_width, shift=shift)
window.plot(model, plot_cols=["HR"], max_subplots=1)
