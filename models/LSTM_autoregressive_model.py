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


class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_output_features)

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        print("inputs.shape:", inputs.shape)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)

        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state, training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions


if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))

    # Create a window and plot it
    window = WindowGenerator(
        input_width=input_width, label_width=label_width, shift=shift
    )

    feedback_model = FeedBack(units=32, out_steps=label_width)

    feedback_model.build(input_shape=(None, input_width, num_input_features))
    print(feedback_model.summary())

    history = compile_and_fit(feedback_model, window)
    # window.plot(feedback_model)

    run_model = tf.function(lambda x: feedback_model(x))
    # This is important, let's fix the input size.
    BATCH_SIZE = 1
    STEPS = 120
    FEATURES = 1
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([BATCH_SIZE, STEPS, FEATURES])
    )

    # Save the model
    feedback_model.save(
        "saved_models/LSTM_autoregressive", save_format="tf", signatures=concrete_func
    )

    # Convert the model to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(
        "saved_models/LSTM_autoregressive"
    )

    # Convert the model
    tflite_model = converter.convert()

    # Save the TFLite model
    with open("saved_models/LSTM_autoregressive/LSTM_autoregressive.tflite", "wb") as f:
        f.write(tflite_model)
