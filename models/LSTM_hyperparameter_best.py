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
    # Create a window and plot it

    units_lstm_1 = 224
    units_lstm_2 = 128
    units_dense1 = 5
    units_dense2 = 20
    units_dense3 = 10

    window = WindowGenerator(
        input_width=input_width, label_width=label_width, shift=shift
    )

    time_series_input = tf.keras.layers.Input(
        shape=(input_width, num_input_features), name="time_series_input"
    )

    x = tf.keras.layers.SimpleRNN(units=units_lstm_1, return_sequences=True)(
        time_series_input
    )
    x = tf.keras.layers.SimpleRNN(units=units_lstm_2, return_sequences=False)(x)

    demographic_input = tf.keras.layers.Input(shape=(7,), name="vector_input")

    demographic_output = tf.keras.layers.Dense(units_dense1)(demographic_input)
    demographic_output = tf.keras.layers.Dense(units_dense2)(demographic_output)
    demographic_output = tf.keras.layers.Dense(units_dense3)(demographic_output)

    concat = tf.keras.layers.concatenate([x, demographic_output])

    output = tf.keras.layers.Dense(128)(concat)
    output = tf.keras.layers.Dense(64)(concat)
    output = tf.keras.layers.Dense(label_width * num_output_features)(concat)
    output = tf.keras.layers.Reshape([label_width, num_output_features])(output)

    multi_lstm_model = tf.keras.models.Model(
        inputs=[time_series_input, demographic_input], outputs=output
    )
    print(multi_lstm_model.summary())

    history = compile_and_fit(multi_lstm_model, window)
    window.plot(multi_lstm_model)

    # Save the model
    multi_lstm_model.save("saved_models/LSTM_hyperparameter_best")
