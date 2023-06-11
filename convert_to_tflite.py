import tensorflow as tf

# Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(
    "saved_models/LSTM_hyperparameter_best"
)

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open(
    "saved_models/LSTM_hyperparameter_best/LSTM_hyperparameter_best.tflite", "wb"
) as f:
    f.write(tflite_model)
