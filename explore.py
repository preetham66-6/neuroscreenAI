import pandas as pd

df = pd.read_csv('parkinsons.csv')

print(df.shape)        # how many rows and columns
print(df.head())       # first 5 rows
print(df.describe())   # statistics of each column
print(df['status'].value_counts())  # how many healthy vs sick
