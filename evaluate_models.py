import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.helpers.window_generator import (
    WindowGenerator,
    num_features,
    input_width,
    label_width,
    shift,
)


# load the model
model = tf.keras.models.load_model("saved_models/LSTM_model")


# Create a window
window = WindowGenerator(input_width=input_width, label_width=label_width, shift=shift)
window.plot(model, max_subplots=10)
