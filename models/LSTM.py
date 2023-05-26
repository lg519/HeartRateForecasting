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

    multi_lstm_model = tf.keras.Sequential(
        [
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(units=32, return_sequences=True),
            tf.keras.layers.LSTM(units=32, return_sequences=False),
            # Shape => [batch, out_steps*features].
            tf.keras.layers.Dense(
                label_width * num_output_features,
                kernel_initializer=tf.initializers.zeros(),
            ),
            # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([label_width, num_output_features]),
        ]
    )

    multi_lstm_model.build(input_shape=(None, input_width, num_input_features))
    print(multi_lstm_model.summary())

    history = compile_and_fit(multi_lstm_model, window)
    # window.plot(feedback_model)

    run_model = tf.function(lambda x: multi_lstm_model(x))
    # This is important, let's fix the input size.
    BATCH_SIZE = 1
    STEPS = 120
    FEATURES = 1
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([BATCH_SIZE, STEPS, FEATURES], multi_lstm_model.inputs[0].dtype)
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
