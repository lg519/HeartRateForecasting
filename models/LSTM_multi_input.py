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

    # Define time series input first
    time_series_input = tf.keras.layers.Input(
        shape=(input_width, num_input_features), name="time_series_input"
    )
    x = tf.keras.layers.LSTM(units=32, return_sequences=True)(time_series_input)
    x = tf.keras.layers.LSTM(units=32, return_sequences=False)(x)
    print(f"x.shape: {x.shape}")

    # Define vector input after time series
    demographic_input = tf.keras.layers.Input(shape=(7,), name="vector_input")

    # Apply Dense layer on demographic input
    demographic_output = tf.keras.layers.Dense(32)(demographic_input)

    # Concatenate LSTM output and vector_input
    concat = tf.keras.layers.concatenate([x, demographic_output])

    # Apply Dense layer on concatenated output
    output = tf.keras.layers.Dense(
        label_width * num_output_features, kernel_initializer=tf.initializers.zeros()
    )(concat)
    output = tf.keras.layers.Reshape([label_width, num_output_features])(output)

    multi_lstm_model = tf.keras.models.Model(
        inputs=[time_series_input, demographic_input], outputs=output
    )

    print(multi_lstm_model.summary())

    history = compile_and_fit(multi_lstm_model, window)
    # window.plot(feedback_model)

    run_model = tf.function(lambda x: multi_lstm_model(x))
    # This is important, let's fix the input size.
    BATCH_SIZE = 128
    STEPS = 120
    FEATURES = 3
    concrete_func = run_model.get_concrete_function(
        [
            tf.TensorSpec(
                [BATCH_SIZE, STEPS, FEATURES], multi_lstm_model.inputs[0].dtype
            ),
            tf.TensorSpec([BATCH_SIZE, 7], dtype=tf.float32),
        ]
    )

    # Save the model
    multi_lstm_model.save(
        "saved_models/LSTM", save_format="tf", signatures=concrete_func
    )

    # Convert the model to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model("saved_models/LSTM")

    # Convert the model
    tflite_model = converter.convert()

    # Save the TFLite model
    with open("saved_models/LSTM/LSTM.tflite", "wb") as f:
        f.write(tflite_model)
