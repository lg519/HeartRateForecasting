from keras_tuner import HyperModel, RandomSearch
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
import json


class HyperModelBuilder(HyperModel):
    def build(self, hp):
        # Define time series input first
        time_series_input = tf.keras.layers.Input(
            shape=(input_width, num_input_features), name="time_series_input"
        )

        # Specify hyperparameters for LSTM layers
        units_lstm_1 = hp.Int("units_lstm_1", min_value=32, max_value=256, step=32)
        units_lstm_2 = hp.Int("units_lstm_2", min_value=32, max_value=256, step=32)

        x = tf.keras.layers.LSTM(units=units_lstm_1, return_sequences=True)(
            time_series_input
        )
        x = tf.keras.layers.LSTM(units=units_lstm_2, return_sequences=False)(x)

        # Define vector input after time series
        demographic_input = tf.keras.layers.Input(shape=(7,), name="vector_input")

        # Apply first Dense layer on demographic input
        units_dense1 = hp.Int("units_dense1", min_value=5, max_value=20, step=5)
        demographic_output = tf.keras.layers.Dense(units_dense1)(demographic_input)

        # Apply second Dense layer
        units_dense2 = hp.Int("units_dense2", min_value=5, max_value=20, step=5)
        demographic_output = tf.keras.layers.Dense(units_dense2)(demographic_output)

        # Apply third Dense layer
        units_dense3 = hp.Int("units_dense3", min_value=5, max_value=20, step=5)
        demographic_output = tf.keras.layers.Dense(units_dense3)(demographic_output)

        # Concatenate LSTM output and vector_input
        concat = tf.keras.layers.concatenate([x, demographic_output])

        # Apply Dense layer on concatenated output
        output = tf.keras.layers.Dense(label_width * num_output_features)(concat)
        output = tf.keras.layers.Reshape([label_width, num_output_features])(output)

        multi_lstm_model = tf.keras.models.Model(
            inputs=[time_series_input, demographic_input], outputs=output
        )

        # Fixed learning rate
        learning_rate = 1e-3

        multi_lstm_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()],
        )

        return multi_lstm_model


# Build the HyperModel
hypermodel = HyperModelBuilder()

# Initialize tuner
tuner = RandomSearch(
    hypermodel,
    objective="val_loss",
    max_trials=15,
    executions_per_trial=1,
    directory="output",
    project_name="hyperparameter_search",
)

# create a window
window = WindowGenerator(input_width=input_width, label_width=label_width, shift=shift)


# Hyperparameter search
tuner.search(window.train, epochs=4, validation_data=window.val)


# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(
    f"The optimal number of units in the LSTM layer is {best_hps.get('units_lstm_1')}"
)
print(
    f"The optimal number of units in the second LSTM layer is {best_hps.get('units_lstm_2')}"
)
print(
    f"The optimal number of neurons in the first dense layer is {best_hps.get('units_dense1')}"
)
print(
    f"The optimal number of neurons in the second dense layer is {best_hps.get('units_dense2')}"
)
print(
    f"The optimal number of neurons in the third dense layer is {best_hps.get('units_dense3')}"
)

# Save the best hyperparameters to a file
with open("best_hyperparameters.json", "w") as f:
    json.dump(best_hps.values, f)

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Save the best model
best_model.save("saved_models/LSTM_hyperparameter_search")
