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
    def __init__(self, units, out_steps, dropout_rate=0.5):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.dropout_rate = dropout_rate
        self.lstm_cells = [
            tf.keras.layers.LSTMCell(units, dropout=self.dropout_rate),
            tf.keras.layers.LSTMCell(units, dropout=self.dropout_rate),
            tf.keras.layers.LSTMCell(units, dropout=self.dropout_rate),
        ]
        self.stacked_lstm_cells = tf.keras.layers.StackedRNNCells(self.lstm_cells)
        # Also wrap the StackedRNNCells in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.stacked_lstm_cells, return_state=True)
        self.dense = tf.keras.layers.Dense(num_output_features)

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *states = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, states

    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, states = self.warmup(inputs)

        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, states = self.stacked_lstm_cells(x, states=states, training=training)
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

    feedback_model = FeedBack(units=128, out_steps=label_width)

    feedback_model.build(input_shape=(None, input_width, num_input_features))
    print(feedback_model.summary())

    history = compile_and_fit(feedback_model, window)
    # window.plot(feedback_model)

    feedback_model.save("saved_models/LSTM_3_layers_model")
