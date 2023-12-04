import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('traffic.csv')

# Convert 'Time' to minutes
df['Time'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.hour * 60 + pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.minute

# Preprocessing columns
numeric_features = ['Time', 'Date', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount']
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

# Create a pipeline with preprocessing and Linear Regression model
model = LinearRegression()

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Use the pipeline to make predictions on the testing set
predictions = pipeline.predict(X_test)

# Evaluate the model performance
mae = mean_absolute_error(y_test, predictions)


print(df.head())
# Display the evaluation metrics
print(f"Mean Absolute Error: {mae}") ##extermely small and means it is very close to actual values in testing set.
