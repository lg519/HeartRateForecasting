import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from .preprocess_data import preprocess_data

df = pd.read_pickle("SportDB.pkl")


(
    train_df,
    val_df,
    test_df,
    train_cat_df,
    val_cat_df,
    test_cat_df,
    hr_scaler,
    br_scaler,
    rr_scaler,
) = preprocess_data(df)

# Set number of output features
num_input_features = 3
num_output_features = 1


# Set input_width
input_width = 120
label_width = 30
shift = 30


class WindowGenerator:
    def __init__(
        self,
        input_width,
        label_width,
        shift,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_cat_df=train_cat_df,
        val_cat_df=val_cat_df,
        test_cat_df=test_cat_df,
        hr_scaler=hr_scaler,
        br_scaler=br_scaler,
        rr_scaler=rr_scaler,
    ):
        # Set random seed
        tf.random.set_seed(1)
        np.random.seed(1)
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.train_cat_df = train_cat_df
        self.val_cat_df = val_cat_df
        self.test_cat_df = test_cat_df

        # Store the scalers
        self.hr_scaler = hr_scaler
        self.br_scaler = br_scaler
        self.rr_scaler = rr_scaler

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

    def split_window(self, features, cat_data):
        # print(f"features shape: {features.shape}")
        # print(f"cat_data shape: {cat_data.shape}")
        # print(f"cat_data: {cat_data}")

        inputs = features[:, self.input_slice, :]

        # Change the last dimention of this vector from 0 to : to plot multi-output sources with window generator
        labels = features[:, self.labels_slice, 0]

        # Add extra dimension to the labels tensor (only needed for uni-output sources)
        labels = tf.expand_dims(labels, axis=-1)

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

        return (inputs, cat_data), labels

    def make_dataset(self, data, cat_data):
        datasets = []

        # print(f"data shape is {len(data[0])}")

        for i, row in enumerate(data):
            row = np.array(row, dtype=np.float32)
            # print(f"row shape: {row.shape}")
            # print(f"row: {row}")
            ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=row,
                targets=None,
                sequence_length=self.total_window_size,
                sequence_stride=1,
                shuffle=False,
                batch_size=128,
            )

            # iterate through all elements of dataset
            # for x in ds:
            #     print(f"element of dataset shape: {x.shape}")

            # print first element of dataset
            # for x in ds.take(1):
            #     print(f"first element of dataset shape: {x.shape}")
            #     print(f"first element of dataset: {x}")

            # Create corresponding dataset for categorical data
            cat_row = cat_data.iloc[i]
            # print(cat_row)
            cat_ts = tf.convert_to_tensor(cat_row)

            # print first element of dataset
            # for x in cat_ds.take(1):
            #     print(f"first element of cat dataset shape: {x.shape}")
            #     print(f"first element of cat dataset: {x}")

            # for x in ds:
            #     print(f"x type is {type(x)}")
            #     print(f"x shape is {x.shape}")

            ds = ds.map(lambda x: (x, tf.broadcast_to(cat_ts, [tf.shape(x)[0], 7])))

            print(ds)

            # iterate through ds and print every element shape
            # for x in ds:
            #     print(f"x[0] shape is {x[0].shape}")
            #     print(f"x[1] shape is {x[1].shape}")
            #     print(x[1])

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

        # print(merged_ds)

        return merged_ds

    def plot(self, model=None, plot_cols=["HR", "BR", "RR"], max_subplots=3):
        (inputs, cat_data), labels = self.example
        print(f"cat_data for these inputs: {cat_data}")
        plt.figure(figsize=(12, 8))

        for n in range(max_subplots):
            for i, plot_col in enumerate(plot_cols):
                plot_col_index = i
                if plot_col_index == 0:
                    scaler = self.hr_scaler
                elif plot_col_index == 1:
                    # Remove break for multi-output plots
                    break
                    scaler = self.br_scaler
                elif plot_col_index == 2:
                    # Remove break for multi-output plots
                    break
                    scaler = self.rr_scaler
                else:
                    raise ValueError(
                        f"plot_col_index must be 0, 1 or 2, not {plot_col_index}"
                    )
                plt.subplot(max_subplots, len(plot_cols), n * len(plot_cols) + i + 1)
                plt.ylabel(f"{plot_col}")
                # print(inputs.shape)
                plt.plot(
                    self.input_indices,
                    scaler.inverse_transform(
                        inputs[n, :, plot_col_index].numpy().reshape(-1, 1)
                    ),
                    label=f"{plot_col} Inputs" if n == 0 else None,
                    marker=".",
                    zorder=-10,
                )

                plt.scatter(
                    self.label_indices,
                    scaler.inverse_transform(
                        labels[n, :, plot_col_index].numpy().reshape(-1, 1)
                    ),
                    edgecolors="k",
                    label=f"{plot_col} Labels" if n == 0 else None,
                    c="#2ca02c",
                    s=64,
                )

                if model is not None:
                    predictions = model(inputs)
                    plt.scatter(
                        self.label_indices,
                        scaler.inverse_transform(
                            predictions[n, :, plot_col_index].numpy().reshape(-1, 1)
                        ),
                        marker="X",
                        edgecolors="k",
                        label=f"{plot_col} Predictions" if n == 0 else None,
                        c="#ff7f0e",
                        s=64,
                    )

                if n == 0:
                    plt.legend()

        plt.xlabel("Time [t]")
        plt.tight_layout()
        plt.show()

    @property
    def train(self):
        return self.make_dataset(self.train_df, self.train_cat_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df, self.val_cat_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df, self.test_cat_df)

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
    print(f"trian_df[0] shape: {train_df[0].shape}")
    window = WindowGenerator(
        input_width=input_width, label_width=label_width, shift=shift
    )

    window.plot()
