import os
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns


import numpy as np
import tensorflow as tf


def read_demographics(file_path):
    with open(file_path, "r") as file:
        next(file)  # Skip the first line with column names
        data = file.read().split()
    return {
        "gender": data[0],
        "age": data[1],
        "weight": data[2],
        "height": data[3],
        "smoking": data[4],
        "alcohol": data[5],
        "weekly_training": data[6],
    }


def read_cardiorespiratory(file_path):
    data = scipy.io.loadmat(file_path, mat_dtype=True)["Data"]
    # print(data["HR"][0, 0].dtype)
    return {
        "ECG": data["ECG"][0, 0],
        "RR": data["RR"][0, 0],
        "BR": data["BR"][0, 0],
        "HR": data["HR"][0, 0],
    }


data_list = []

for sport in os.listdir("SportDB"):
    sport_path = os.path.join("SportDB", sport)
    for subject in os.listdir(sport_path):
        subject_path = os.path.join(sport_path, subject)
        for acquisition in os.listdir(subject_path):
            acquisition_path = os.path.join(subject_path, acquisition)
            demographic_path = os.path.join(acquisition_path, "Dem.txt")
            cardiorespiratory_path = os.path.join(acquisition_path, "Data.mat")

            demographics = read_demographics(demographic_path)
            cardiorespiratory = read_cardiorespiratory(cardiorespiratory_path)

            data = {
                **demographics,
                **cardiorespiratory,
                "sport": sport,
                "subject": subject,
                "acquisition": acquisition,
            }
            data_list.append(data)


df = pd.DataFrame(data_list)


# save the dataframe to a csv file
# df.to_pickle("SportDB.pkl")


demographics_columns = [
    "gender",
    "age",
    "weight",
    "height",
    "smoking",
    "alcohol",
    "weekly_training",
]

# Convert columns to numeric types
for column in demographics_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Visualize demographic data using Seaborn's pairplot
sns.pairplot(
    df[demographics_columns].dropna()
)  # Remove rows with NaN values for plotting
plt.show()


heart_rate_data = df["HR"]
sample_index = 67  # Change this value to select a different sample
heart_rate_sample = heart_rate_data.iloc[sample_index]

breathing_rate_data = df["BR"]
breathing_rate_sample = breathing_rate_data.iloc[sample_index]

ecg_data = df["ECG"]
ecg_sample = ecg_data.iloc[sample_index]

rr_data = df["RR"]
rr_sample = rr_data.iloc[sample_index]

## Create a subplot with 2 rows and 2 columns
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot 1: Heart Rate
axs[0, 0].plot(heart_rate_sample)
axs[0, 0].set_title("Heart Rate")
axs[0, 0].set_xlabel("Sample")
axs[0, 0].set_ylabel("Rate")

# Plot 2: ECG
axs[0, 1].plot(ecg_sample)
axs[0, 1].set_title("ECG")
axs[0, 1].set_xlabel("Sample")
axs[0, 1].set_ylabel("Value")

# Plot 3: Breathing Rate
axs[1, 0].plot(breathing_rate_sample)
axs[1, 0].set_title("Breathing Rate")
axs[1, 0].set_xlabel("Sample")
axs[1, 0].set_ylabel("Rate")

# Plot 4: RR Interval
axs[1, 1].plot(rr_sample)
axs[1, 1].set_title("RR Interval")
axs[1, 1].set_xlabel("Sample")
axs[1, 1].set_ylabel("Interval")

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()
