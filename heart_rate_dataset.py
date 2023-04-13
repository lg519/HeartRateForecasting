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


# demographics_columns = [
#     "gender",
#     "age",
#     "weight",
#     "height",
#     "smoking",
#     "alcohol",
#     "weekly_training",
# ]

# Convert columns to numeric types
# for column in demographics_columns:
#     df[column] = pd.to_numeric(df[column], errors="coerce")

# # Visualize demographic data using Seaborn's pairplot
# sns.pairplot(
#     df[demographics_columns].dropna()
# )  # Remove rows with NaN values for plotting
# plt.show()


heart_rate_data = df["HR"]
sample_index = 0  # Change this value to select a different sample
heart_rate_sample = heart_rate_data.iloc[sample_index]

breathing_rate_data = df["BR"]
breathing_rate_sample = breathing_rate_data.iloc[sample_index]


# Plot the heart rate and breathing rate time series
plt.plot(heart_rate_sample)

plt.plot(breathing_rate_sample)
plt.legend(["Heart rate", "Breathing rate"])
plt.show()
