import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# Function to check if there's any nested array with a 0
def has_zero(arr):
    for nested_arr in arr:
        if nested_arr[0] == 0:
            return True
    return False


def preprocess_data(df):
    # Extract heart rate data
    heart_rate_data = df["HR"]
    # Extract breathing rate data
    breathing_rate_data = df["BR"]

    # Apply the has_zero function to each heart rate row
    zero_check_hr = heart_rate_data.apply(has_zero)

    # Filter the rows based on the check results
    heart_rate_data = heart_rate_data[~zero_check_hr].reset_index(drop=True)
    breathing_rate_data = breathing_rate_data[~zero_check_hr].reset_index(drop=True)

    # Scale the data between -1 and 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    heart_rate_data = [scaler.fit_transform(hr) for hr in heart_rate_data]
    breathing_rate_data = [scaler.fit_transform(br) for br in breathing_rate_data]

    # Combine HR and BR data
    combined_data = [
        np.column_stack((hr, br))
        for hr, br in zip(heart_rate_data, breathing_rate_data)
    ]

    # Split the data into train, validation, and test sets
    n = len(combined_data)
    train_df = combined_data[0 : int(n * 0.7)]
    val_df = combined_data[int(n * 0.7) : int(n * 0.9)]
    test_df = combined_data[int(n * 0.9) :]

    return train_df, val_df, test_df


def plot_samples(train_df, num_samples=3):
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 6))
    fig.subplots_adjust(hspace=0.5)

    for i, ax in enumerate(axes):
        sample = train_df[i]
        ax.plot(sample[:, 0], label="HR")
        ax.plot(sample[:, 1], label="BR")
        ax.set_title(f"Sample {i+1}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()

    plt.show()


if __name__ == "__main__":
    # Example usage
    df = pd.read_pickle("SportDB.pkl")
    train_df, val_df, test_df = preprocess_data(df)
    plot_samples(train_df)
