import tensorflow as tf
from helpers.window_generator import (
    WindowGenerator,
    input_width,
    label_width,
    shift,
)


class Baseline(tf.keras.Model):
    def __init__(self, label_width):
        super().__init__()
        self.label_width = label_width

    def call(self, inputs):
        last_value = inputs[:, -1:, 0]  # Get the last value of the HR data
        repeated_last_value = tf.repeat(last_value, self.label_width, axis=1)
        return tf.expand_dims(repeated_last_value, axis=-1)


baseline_window = WindowGenerator(
    input_width=input_width, label_width=label_width, shift=shift
)


if __name__ == "__main__":
    # Create the baseline model
    baseline_model = Baseline(label_width=baseline_window.label_width)

    baseline_model.compile(loss=tf.losses.MeanAbsoluteError())

    # # Evaluate the performance of the baseline model on test data
    test_performance = {}
    test_performance["Baseline"] = baseline_model.evaluate(baseline_window.test)

    # Visualize the predictions of the baseline model
    baseline_window.plot(baseline_model)

    # Save the baseline model
    baseline_model.save("saved_models/baseline")
