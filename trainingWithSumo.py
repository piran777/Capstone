import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# Load your machine learning dataset
df = pd.read_csv('Traffic.csv')

# Load the trafficImageDetection data
traffic_image_data = pd.read_csv('trafficImageDetected.csv')

# Feature Engineering: Extract hour and minute from the 'Time' column
df['Hour'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.hour
df['Minute'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.minute

traffic_image_data['Hour'] = pd.to_datetime(traffic_image_data['Time'], format='%I:%M:%S %p').dt.hour
traffic_image_data['Minute'] = pd.to_datetime(traffic_image_data['Time'], format='%I:%M:%S %p').dt.minute

df.drop(['Time', 'Date'], axis='columns', inplace=True)

# Combine traffic image detection data with machine learning data
combined_data = pd.concat([df, traffic_image_data], axis=0, ignore_index=True)

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

# Update your preprocessor with an imputation step
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values
            ('scaler', StandardScaler())]), numeric_features),
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

# Use the pipeline to predict the 'Traffic Situation' for the combined data
combined_predictions = pipeline.predict(X_combined)
print("Predictions for combined data:\n", combined_predictions)
