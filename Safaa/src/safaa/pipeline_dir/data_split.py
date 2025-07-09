import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('data/preprocessed_copyrights.csv')

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save to CSV files
train_df.to_csv('data/train_data.csv', index=False)
test_df.to_csv('data/test_data.csv', index=False)

print("✅ Split complete: train_data.csv and test_data.csv created.")
