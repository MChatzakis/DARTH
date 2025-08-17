# Script to average the running times of experiments

import os
import pandas as pd
import sys

def cast_int_columns(df):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):  # Check if the column is numeric
            if (df[col] % 1 == 0).all():  # Check if all values are integers
                df[col] = df[col].astype(int)
    return df

input_file_prefix = sys.argv[1]
output_file = sys.argv[2]
experiment_times = int(sys.argv[3])

all_dfs = []
for time in range(1, experiment_times + 1):
    current_df = pd.read_csv(f"{input_file_prefix}_t{time}.txt")
    all_dfs.append(current_df)

# Concatenate all dataframes
combined_df = pd.concat(all_dfs)

# Group by 'query-id' and compute the mean for each group
mean_df = combined_df.groupby('qid').mean().reset_index()
mean_df = cast_int_columns(mean_df)

# Print log
#print(f"Averaged running times from {experiment_times} experiments")

# Save the averaged dataframe to a CSV file
mean_df.to_csv(output_file, index=False)

