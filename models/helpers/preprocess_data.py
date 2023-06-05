import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.signal import resample


# Function to check if there's any nested array with a 0
def filter_data(arr):
    for nested_arr in arr:
        # If the nested array contains a 0 or a number greater than 1000, return True
        if nested_arr[0] == 0 or nested_arr[0] > 1000:
            return True

    return False


def preprocess_data(df):
    # Extract categorical data
    cat_cols = [
        "gender",
        "age",
        "weight",
        "height",
        "smoking",
        "alcohol",
        "weekly_training",
    ]
    cat_data = df[cat_cols]

    # Replace 'NA' strings with numpy NaN
    cat_data = cat_data.replace("NA", np.nan)

    # Apply mode imputation to deal with missing values
    cat_data.fillna(cat_data.mode().iloc[0], inplace=True)

    # Fill missing values using mode imputation
    for col in cat_cols:
        cat_data[col].fillna(cat_data[col].mode()[0])

    # Convert data to float
    cat_data = cat_data.astype(float)

    print(cat_data.head())
    # Extract heart rate data
    heart_rate_data = df["HR"]
    # print(f"first sample of HR data shape is {df['HR'][0].shape}")
    # Extract breathing rate data
    breathing_rate_data = df["BR"]
    # print(f"first sample of BR data shape is {df['BR'][0].shape}")
    # Extract ECG data
    ecg_data = df["ECG"]
    # print(f"first sample of ECG data shape is {df['ECG'][0].shape}")
    # Extract RR data
    rr_data = df["RR"]
    # print(f"first sample of RR data shape is {df['RR'][0].shape}")

    print(len(heart_rate_data))

    # Filter the rows
    zero_check_hr = heart_rate_data.apply(filter_data)
    heart_rate_data = heart_rate_data[~zero_check_hr].reset_index(drop=True)
    breathing_rate_data = breathing_rate_data[~zero_check_hr].reset_index(drop=True)
    # ecg_data = ecg_data[~zero_check_hr].reset_index(drop=True)
    rr_data = rr_data[~zero_check_hr].reset_index(drop=True)
    print(len(heart_rate_data))

    zero_check_br = breathing_rate_data.apply(filter_data)
    heart_rate_data = heart_rate_data[~zero_check_br].reset_index(drop=True)
    breathing_rate_data = breathing_rate_data[~zero_check_br].reset_index(drop=True)
    # ecg_data = ecg_data[~zero_check_br].reset_index(drop=True)
    rr_data = rr_data[~zero_check_br].reset_index(drop=True)
    print(len(heart_rate_data))

    zero_check_rr = rr_data.apply(filter_data)
    heart_rate_data = heart_rate_data[~zero_check_rr].reset_index(drop=True)
    breathing_rate_data = breathing_rate_data[~zero_check_rr].reset_index(drop=True)
    # ecg_data = ecg_data[~zero_check_rr].reset_index(drop=True)
    rr_data = rr_data[~zero_check_rr].reset_index(drop=True)
    print(len(heart_rate_data))

    # Concatenate each row of HR and BR data
    concatenated_hr = np.concatenate(heart_rate_data)
    concatenated_br = np.concatenate(breathing_rate_data)
    concatenated_rr = np.concatenate(rr_data)

    # Scale the data
    hr_scaler = MinMaxScaler()
    br_scaler = MinMaxScaler()
    rr_scaler = MinMaxScaler()

    # Fit the scaler to the concatenated data
    hr_scaler.fit(concatenated_hr)
    br_scaler.fit(concatenated_br)
    rr_scaler.fit(concatenated_rr)

    # Print minimum and maximum value of hr_scaler
    # print("Minimum value hr_scaler:", hr_scaler.data_min_)
    # print("Maximum value hr_scaler:", hr_scaler.data_max_)

    # Transform the data
    heart_rate_data = [hr_scaler.transform(hr) for hr in heart_rate_data]
    breathing_rate_data = [br_scaler.transform(br) for br in breathing_rate_data]
    rr_data = [rr_scaler.transform(rr) for rr in rr_data]

    # Select first 150 samples
    # hr_selected = heart_rate_data[1]
    # br_selected = breathing_rate_data[1]
    # rr_selected = rr_data[1]

    # # Plotting each array
    # plt.figure(figsize=(14, 10))

    # plt.subplot(3, 1, 1)
    # plt.plot(hr_selected)
    # plt.title("Heart Rate Data")

    # plt.subplot(3, 1, 2)
    # plt.plot(br_selected)
    # plt.title("Breathing Rate Data")

    # plt.subplot(3, 1, 3)
    # plt.plot(rr_selected)
    # plt.title("RR Data")

    # plt.tight_layout()
    # plt.show()

    # Combine HR and BR data
    combined_data = [
        np.column_stack((hr, br, rr))
        for hr, br, rr in zip(heart_rate_data, breathing_rate_data, rr_data)
    ]

    # Only use Heart Rate data. TODO: use also other data for experiments in the report
    # combined_data = heart_rate_data

    # Split the cardiorespiratory data into train, validation, and test sets
    n = len(combined_data)
    train_df = combined_data[0 : int(n * 0.7)]
    val_df = combined_data[int(n * 0.7) : int(n * 0.9)]
    test_df = combined_data[int(n * 0.9) :]

    # Split the categorical data into train, validation, and test sets
    n = len(cat_data)
    train_cat_data = cat_data[0 : int(n * 0.7)]
    val_cat_data = cat_data[int(n * 0.7) : int(n * 0.9)]
    test_cat_data = cat_data[int(n * 0.9) :]

    return (
        train_df,
        val_df,
        test_df,
        train_cat_data,
        val_cat_data,
        test_cat_data,
        hr_scaler,
        br_scaler,
        rr_scaler,
    )


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


def plot_samples_with_cat_data(train_df, train_cat_data, num_samples=2):
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 6))
    fig.subplots_adjust(hspace=0.5)

    for i in range(num_samples):
        # Plot time-series data
        ax1 = axes[i, 0]
        sample = train_df[i]
        ax1.plot(sample[:, 0], label="HR")
        ax1.plot(sample[:, 1], label="BR")
        ax1.plot(sample[:, 2], label="RR")
        ax1.set_title(f"Sample {i+1}")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Value")
        ax1.legend()

        # Plot categorical data
        ax2 = axes[i, 1]
        cat_sample = train_cat_data.iloc[i]
        ax2.barh(cat_sample.index, cat_sample.values)
        ax2.set_title(f"Sample {i+1} Demographic Data")
        ax2.set_xlabel("Value")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    df = pd.read_pickle("SportDB.pkl")
    print(df.columns)
    (
        train_df,
        val_df,
        test_df,
        train_cat_data,
        val_cat_data,
        test_cat_data,
        _,
        _,
        _,
    ) = preprocess_data(df)
    print(len(train_df))
    print(train_df[0].shape)
    # plot_samples(train_df)
    plot_samples_with_cat_data(train_df, train_cat_data)
