import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from helpers.window_generator import (
    WindowGenerator,
    num_features,
    input_width,
    label_width,
    shift,
)
from helpers.compile_and_fit import compile_and_fit


class StackedLSTM(tf.keras.layers.Layer):
    def __init__(self, units, num_layers, dropout_rate):
        super().__init__()
        self.units = units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.dropout_layers = [
            tf.keras.layers.Dropout(dropout_rate) for _ in range(num_layers)
        ]
        self.lstm_layers = [
            tf.keras.layers.LSTM(
                units,
                return_sequences=True,
                return_state=True,
            )
            for _ in range(num_layers - 1)
        ] + [
            tf.keras.layers.LSTM(
                units,
                return_state=True,
            )
        ]

    def call(self, inputs, training=None, states=None):
        if states is None:
            states = [None] * self.num_layers

        x = inputs
        new_states = []
        for i, (dropout, lstm) in enumerate(zip(self.dropout_layers, self.lstm_layers)):
            x = dropout(x, training=training)
            x, h, c = lstm(x, initial_state=states[i], training=training)
            new_states.append((h, c))

        return x, new_states


class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps, num_layers=3, dropout_rate=0.5):
        super().__init__()
        self.out_steps = out_steps
        self.stacked_lstm = StackedLSTM(units, num_layers, dropout_rate)
        self.dense = tf.keras.layers.Dense(num_features)

    def warmup(self, inputs):
        x, states = self.stacked_lstm(inputs)
        prediction = self.dense(x)
        return prediction, states

    def call(self, inputs, training=None):
        predictions = []
        prediction, states = self.warmup(inputs)
        predictions.append(prediction)

        for n in range(1, self.out_steps):
            x = prediction[:, -1:, np.newaxis]
            x, states = self.stacked_lstm(x, training=training, states=states)
            prediction = self.dense(x)
            predictions.append(prediction)

        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions


if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))

    window = WindowGenerator(
        input_width=input_width, label_width=label_width, shift=shift
    )

    feedback_model = FeedBack(units=128, out_steps=label_width)

    feedback_model.build(input_shape=(None, input_width, num_features))
    print(feedback_model.summary())

    history = compile_and_fit(feedback_model, window)

    feedback_model.save("models/LSTM_3_layers_cuda_model")
