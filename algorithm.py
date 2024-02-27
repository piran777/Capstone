import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Traffic.csv')

# Convert 'Time' to minutes

# Feature Engineering: Extract hour and minute from the 'Time' column
df['Hour'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.hour
df['Minute'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.minute

df.drop(['Time', 'Date'], axis='columns', inplace=True)
# Preprocessing columns
numeric_features = ['Hour', 'Minute', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount']
categorical_features = ['Day of the week']

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),  # Include 'Time' in numeric features
        ('cat', OneHotEncoder(), categorical_features)
    ])


X = df.drop("Total", axis='columns')  # Features
y = df["Total"]  # Target variable

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])


# Cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
cv_mae = -cv_scores.mean()

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)


# Use the pipeline to make predictions on the testing set
predictions = pipeline.predict(X_test)

# Evaluate the model performance
mae = mean_absolute_error(y_test, predictions)


print(df.head())
# Display the evaluation metrics

# Display evaluation metrics
print(f"Cross-validated MAE: {cv_mae}")
print(f"MAE on Test Set: {mae}") ##extermely small and means it is very close to actual values in testing set.
# Create a pivot table to rearrange the data for the heat map
pivot_table = df.pivot_table(values='Total', index='Time', columns='Day of the week')

# Create a larger heat map using seaborn
plt.figure(figsize=(24, 20))  # Adjust the figsize here
sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt='g', linewidths=.5, cbar_kws={'label': 'Traffic Density'})
plt.title('Traffic Density Heat Map')
plt.show()
