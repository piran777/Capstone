import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Load your machine learning dataset
df = pd.read_csv('Traffic.csv')

# Assuming 'sumo_data' is a DataFrame with features similar to your original dataset
sumo_data = pd.DataFrame({
    'Hour': [1, 1],
    'Minute': [0, 15],
    'Day of the week': ['Wednesday', 'Wednesday'],
    'CarCount': [30, 25],
    'BikeCount': [2, 1],
    'BusCount': [5, 3],
    'TruckCount': [8, 6],
    'Total': [39, 28]  # Assuming you have the 'Total' column in your SUMO data
})

# Feature Engineering: Extract hour and minute from the 'Time' column
df['Hour'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.hour
df['Minute'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.minute

df.drop(['Time', 'Date'], axis='columns', inplace=True)

# Combine SUMO data with machine learning data
combined_data = pd.concat([df, sumo_data], axis=0, ignore_index=True)

# Convert 'Traffic Situation' to numeric labels
label_encoder = LabelEncoder()
combined_data['Traffic Situation'] = label_encoder.fit_transform(combined_data['Traffic Situation'])

# Define features and target for machine learning
X_combined = combined_data[['Hour', 'Minute', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total']]
y_combined = combined_data['Traffic Situation']

# Split the combined data
X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Define preprocessor
numeric_features = ['Hour', 'Minute', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total']
categorical_features = []  # Remove 'Day of the week' from categorical features

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Create pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Cross-validation
cv_scores = cross_val_score(pipeline, X_combined, y_combined, cv=5, scoring='accuracy')
cv_accuracy = cv_scores.mean()

# Fit the pipeline on the training data
pipeline.fit(X_train_combined, y_train_combined)

# Use the pipeline to make predictions on the testing set
predictions = pipeline.predict(X_test_combined)

# Evaluate the model performance
accuracy = accuracy_score(y_test_combined, predictions)
classification_rep = classification_report(y_test_combined, predictions)

# Display evaluation metrics
print(f"Cross-validated Accuracy: {cv_accuracy}")
print(f"Accuracy on Test Set: {accuracy}")
print("Classification Report on Test Set:\n", classification_rep)

# Use the pipeline to predict the 'Traffic Situation' for your SUMO data
sumo_predictions = pipeline.predict(sumo_data[['Hour', 'Minute', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total']])
print("Predictions for SUMO data:\n", sumo_predictions)
