import tensorflow as tf
from tensorflow import keras
from helpers.window_generator import (
    WindowGenerator,
    num_features,
    input_width,
    label_width,
    shift,
)


window = WindowGenerator(input_width=input_width, label_width=label_width, shift=shift)


# Load the saved Keras model
loaded_model = keras.models.load_model("models/LSTM_autoregressive_model")

# Assuming you have 'train', 'val', and 'test' datasets ready as in the provided code
# Replace 'lstm_window' with the appropriate window object for your datasets
train_performance = {}
train_performance["LSTM"] = loaded_model.evaluate(window.train)

val_performance = {}
val_performance["LSTM"] = loaded_model.evaluate(window.val)

test_performance = {}
test_performance["LSTM"] = loaded_model.evaluate(window.test)
