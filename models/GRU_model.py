import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from helpers.window_generator import (
    WindowGenerator,
    num_output_features,
    num_input_features,
    input_width,
    label_width,
    shift,
)
from helpers.compile_and_fit import compile_and_fit


if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))

    # Create a window and plot it
    window = WindowGenerator(
        input_width=input_width, label_width=label_width, shift=shift
    )

    multi_gru_model = tf.keras.Sequential(
        [
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.GRU(units=32, return_sequences=True),
            tf.keras.layers.GRU(units=32, return_sequences=False),
            # Shape => [batch, out_steps*features].
            tf.keras.layers.Dense(
                label_width * num_output_features,
                kernel_initializer=tf.initializers.zeros(),
            ),
            # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([label_width, num_output_features]),
        ]
    )

    multi_gru_model.build(input_shape=(None, input_width, num_input_features))
    print(multi_gru_model.summary())

    history = compile_and_fit(multi_gru_model, window)
    # window.plot(feedback_model)

    multi_gru_model.save("saved_models/GRU_model")
