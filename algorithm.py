import pandas as pd

# Assuming the TXT file is tab-separated
df = pd.read_csv('traffic.csv')
# Display the first few rows of the DataFrame (To test if its working)
print(df.head())