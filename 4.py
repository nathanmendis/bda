import pandas as pd

# Load dataset
df = pd.read_csv("weather.csv")

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Filter records for 2013
df_2013 = df[df['date'].dt.year == 2013]

# Find row with maximum snowfall
max_snowfall_row = df_2013.loc[df_2013['snowfall'].idxmax()]

# Display result
print("Day with maximum snowfall:")
print(max_snowfall_row['date'])

print("Station:")
print(max_snowfall_row['station'])

print("Maximum snowfall:")
print(max_snowfall_row['snowfall'])