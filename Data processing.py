import pandas as pd
import numpy as np

# Load the data from the provided CSV file
file_path = r"File address"
data = pd.read_csv(file_path)

# Ensure the 'Date' column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Sort the data by Date to ensure we have the correct order
data.sort_values(by=['Product Type', 'Date'], inplace=True)

# Calculate 7-Day Mean (rolling window of 7 days)
data['7-Day Mean'] = data.groupby('Product Type')['market price'].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)

# Calculate 30-Day Range (rolling window of 30 days)
data['30-Day Range'] = data.groupby('Product Type')['market price'].rolling(window=30, min_periods=1).apply(lambda x: x.max() - x.min()).reset_index(level=0, drop=True)

# Calculate Normalized Price (normalized by the maximum price in the entire period for each product)
data['Normalized Price'] = data.groupby('Product Type')['market price'].transform(lambda x: x / x.max())

# Save the updated dataframe to a new CSV file
output_file_path = "Final data.csv"
data.to_csv(output_file_path, index=False)

# Show the first few rows of the updated dataframe
print(data.head())
