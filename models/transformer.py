import tensorflow as tf
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
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=res.shape[-1], kernel_size=1)(x)
    x = layers.Dropout(dropout)(x)
    return x + res


def get_positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    positional_encoding = np.zeros((seq_len, d_model))
    positional_encoding[:, 0::2] = np.sin(pos * div_term)
    positional_encoding[:, 1::2] = np.cos(pos * div_term)
    return tf.constant(positional_encoding, dtype=tf.float32)


def create_transformer_model(
    input_shape, num_layers, head_size, num_heads, ff_dim, dropout=0.1
):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    seq_len = input_shape[0]
    d_model = input_shape[1]

    # Add positional encoding
    x = x + get_positional_encoding(seq_len, d_model)

    for _ in range(num_layers):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(num_output_features, activation="linear")(x)

    model = Model(inputs=inputs, outputs=x)
    return model


# Model parameters
num_layers = 4
head_size = 64
num_heads = 5
ff_dim = 128
dropout = 0.1

# Create and compile the model
model = create_transformer_model(
    input_shape=(input_width, num_input_features),
    num_layers=num_layers,
    head_size=head_size,
    num_heads=num_heads,
    ff_dim=ff_dim,
    dropout=dropout,
)

model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.MeanAbsoluteError()],
)

# Create the WindowGenerator instance
window = WindowGenerator(input_width=input_width, label_width=label_width, shift=shift)

model.summary()

# Train the model
history = model.fit(
    window.train,
    epochs=10,
    validation_data=window.val,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)],
)

# Evaluate the model
model.evaluate(window.test)

# Plot the predictions
window.plot(model)
