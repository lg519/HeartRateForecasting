import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# load the data
# df = pd.read_csv("SportDB.csv")

df = pd.read_pickle("SportDB.pkl")
# extract heart rate data
heart_rate_data = df["HR"]

# Split the data into train, validation, and test sets
n = len(heart_rate_data)
train_df = heart_rate_data[0 : int(n * 0.7)]
val_df = heart_rate_data[int(n * 0.7) : int(n * 0.9)]
test_df = heart_rate_data[int(n * 0.9) :]

num_features = 1


# Extract heart rate data (assuming each entry is an array of readings)
train_df_data_concatenated = np.concatenate(train_df.values)


# Preprocess the data
train_mean = train_df_data_concatenated.mean()
train_std = train_df_data_concatenated.std()


train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# Set input_width
input_width = 50
label_width = 10
shift = 10


class WindowGenerator:
    def __init__(
        self,
        input_width,
        label_width,
        shift,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_columns=None,
    ):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        # self.label_columns = label_columns
        # if label_columns is not None:
        #     self.label_columns_indices = {
        #         name: i for i, name in enumerate(label_columns)
        #     }
        # self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
            ]
        )

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        # if self.label_columns is not None:
        #     labels = tf.stack(
        #         [
        #             labels[:, :, self.column_indices[name]]
        #             for name in self.label_columns
        #         ],
        #         axis=-1,
        #     )

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        datasets = []

        for row in data:
            row = np.array(row, dtype=np.float32)
            # print(f"row shape: {row.shape}")
            # print(f"row: {row}")
            ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=row,
                targets=None,
                sequence_length=self.total_window_size,
                sequence_stride=1,
                shuffle=True,
                batch_size=32,
            )
            # print first element of dataset
            # for x in ds.take(1):
            #     print(f"first element of dataset shape: {x.shape}")
            #     print(f"first element of dataset: {x}")

            ds = ds.map(self.split_window)

            # print first element of dataset after split
            # for x, y in ds.take(1):
            #     print(f"first element of dataset after split shape: {x.shape}")
            #     print(f"first element of dataset after split: {x}")
            #     print(f"first label of dataset after split shape: {y.shape}")
            #     print(f"first label of dataset after split: {y}")

            datasets.append(ds)

        merged_ds = datasets[0]
        for ds in datasets[1:]:
            merged_ds = merged_ds.concatenate(ds)

        print(merged_ds)

        return merged_ds

    def plot(self, model=None, plot_col="HR", max_subplots=3):
        inputs, labels = self.example
        # print(f"inputs shape: {inputs.shape}")
        # print(f"labels shape: {labels.shape}")
        plt.figure(figsize=(12, 8))
        plot_col_index = 0  # Assuming 'HR' is the only column

        for n in range(max_subplots):
            plt.subplot(max_subplots, 1, n + 1)
            plt.ylabel(f"{plot_col} ")
            plt.plot(
                self.input_indices,
                inputs[n, :, plot_col_index],
                label="Inputs",
                marker=".",
                zorder=-10,
            )

            plt.scatter(
                self.label_indices,
                labels[n, :, plot_col_index],
                edgecolors="k",
                label="Labels",
                c="#2ca02c",
                s=64,
            )
            if model is not None:
                predictions = model(inputs)
                # print(f"predictions shape: {predictions.shape}")
                plt.scatter(
                    self.label_indices,
                    predictions[n, :, plot_col_index],
                    marker="X",
                    edgecolors="k",
                    label="Predictions",
                    c="#ff7f0e",
                    s=64,
                )

            if n == 0:
                plt.legend()

        plt.xlabel("Time [t]")
        plt.show()

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, "_example", None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


if __name__ == "__main__":
    # visualize the data
    plt.plot(train_df_data_concatenated)
    plt.show()

    print(f"trian_df shape: {train_df[0].shape}")
    window = WindowGenerator(
        input_width=input_width, label_width=label_width, shift=shift
    )

    window.plot()
