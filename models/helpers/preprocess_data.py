import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# Function to check if there's any nested array with a 0
def filter_data(arr):
    for nested_arr in arr:
        # If the nested array contains a 0 or a number greater than 1000, return True
        if nested_arr[0] == 0 or nested_arr[0] > 1000:
            return True

    return False


def preprocess_data(df):
    # Extract heart rate data
    heart_rate_data = df["HR"]
    print(f"first sample of HR data shape is {df['HR'][0].shape}")
    # Extract breat hing rate data
    breathing_rate_data = df["BR"]
    print(f"first sample of BR data shape is {df['BR'][0].shape}")
    # Extract ECG data
    ecg_data = df["ECG"]
    print(f"first sample of ECG data shape is {df['ECG'][0].shape}")
    # Extract RR data
    rr_data = df["RR"]
    print(f"first sample of RR data shape is {df['RR'][0].shape}")

    # print(len(heart_rate_data))

    # Filter the rows
    zero_check_hr = heart_rate_data.apply(filter_data)
    heart_rate_data = heart_rate_data[~zero_check_hr].reset_index(drop=True)
    breathing_rate_data = breathing_rate_data[~zero_check_hr].reset_index(drop=True)
    # print(len(heart_rate_data))

    zero_check_br = breathing_rate_data.apply(filter_data)
    heart_rate_data = heart_rate_data[~zero_check_br].reset_index(drop=True)
    breathing_rate_data = breathing_rate_data[~zero_check_br].reset_index(drop=True)
    # print(len(heart_rate_data))

    # Concatenate each row of HR and BR data
    concatenated_hr = np.concatenate(heart_rate_data)
    concatenated_br = np.concatenate(breathing_rate_data)
    # print(concatenated_br)

    # Plot the concatenated data
    # plt.plot(concatenated_br)
    # plt.show()

    # Scale the data
    hr_scaler = MinMaxScaler()
    br_scaler = MinMaxScaler()

    # Fit the scaler to the concatenated data
    hr_scaler.fit(concatenated_hr)
    br_scaler.fit(concatenated_br)

    # Transform the data
    heart_rate_data = [hr_scaler.transform(hr) for hr in heart_rate_data]
    breathing_rate_data = [br_scaler.transform(br) for br in breathing_rate_data]

    # Combine HR and BR data
    combined_data = [
        np.column_stack((hr, br))
        for hr, br in zip(heart_rate_data, breathing_rate_data)
    ]

    # Only use Heart Rate data. TODO: use also other data for experiments in the report
    combined_data = heart_rate_data

    # Split the data into train, validation, and test sets
    n = len(combined_data)
    train_df = combined_data[0 : int(n * 0.7)]
    val_df = combined_data[int(n * 0.7) : int(n * 0.9)]
    test_df = combined_data[int(n * 0.9) :]

    return train_df, val_df, test_df, hr_scaler, br_scaler


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
    train_df, val_df, test_df, _, _ = preprocess_data(df)
    print(len(train_df))
    print(train_df[0].shape)
    # plot_samples(train_df)
