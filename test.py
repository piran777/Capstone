import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import folium 
from folium.plugins import HeatMap

import numpy as np
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

# Update your preprocessor with an imputation step for 'Total' column
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values for numeric features
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

print(X_test_combined.columns)

traffic_lights_df = pd.read_csv('TrafficLightsLocations.csv')

# Convert 'Latitude' and 'Longitude' from DMS format to decimal
import re

def dms2dec(dms_str):
    dms_str = re.sub(r'\s', '', dms_str)
    sign = -1 if re.search('[SWsw]', dms_str) else 1

    parts = re.split('[°\'"]', dms_str)
    degrees, minutes, seconds = map(float, parts[:3])
    return sign * (degrees + minutes / 60 + seconds / 3600)


# apply dms2dec to 'Latitude' and 'Longitude' columns
traffic_lights_df['Latitude'] = traffic_lights_df['Latitude'].apply(dms2dec)
traffic_lights_df['Longitude'] = traffic_lights_df['Longitude'].apply(dms2dec)

#Export to csv for handling in decimal format
traffic_lights_df.to_csv('TrafficLightsLocations_decimal.csv', index=False)

# Assign 'Latitude' and 'Longitude' to combined_data
combined_data['Latitude'] = traffic_lights_df['Latitude']
combined_data['Longitude'] = traffic_lights_df['Longitude']


pipeline.fit(X_train_combined, y_train_combined)


# Use the pipeline to make predictions on the testing set
predictions = pipeline.predict(X_test_combined)                             

# Create a folium map
# Create a folium map
m = folium.Map(location=[combined_data['Latitude'].mean(), combined_data['Longitude'].mean()], zoom_start=13)

# Drop rows with NaN values in 'Latitude' or 'Longitude' columns
combined_data = combined_data.dropna(subset=['Latitude', 'Longitude'])

# Create a HeatMap layer using the latitude, longitude, and Total columns
heat_data = [[point[0], point[1], weight] for point, weight in zip(zip(combined_data['Latitude'], combined_data['Longitude']), combined_data['Total'])]

HeatMap(heat_data).add_to(m)


# Save the map to an HTML file
m.save("traffic_map.html")



print(X_combined.head())
def get_decision_for_row(row_index):
    # Extract features for the requested row
    row_data = X_combined.iloc[[row_index]]
    
    # Use the trained pipeline to predict the traffic situation for the requested row
    predicted_traffic_situation = pipeline.predict(row_data)[0]
    
    print("Predicted Traffic Situation Label:", predicted_traffic_situation)
    
    # Decode the predicted traffic situation label
    predicted_traffic_situation = label_encoder.inverse_transform([predicted_traffic_situation])[0]
    
    # Make decision based on the traffic situation
    decision = ""
    if predicted_traffic_situation == 'heavy':  # Adjusted to lowercase
        decision = "Implement strict traffic control measures."
    elif predicted_traffic_situation == 'normal':  # Adjusted to lowercase
        decision = "Monitor traffic closely and consider adjusting traffic signals if necessary."
    elif predicted_traffic_situation == 'low':  # Adjusted to lowercase
        decision = "No immediate action required."
    
    return predicted_traffic_situation, decision
import webbrowser
#webbrowser.open("traffic_map.html")
# Example usge:
row_index = 17  # Change this to any row index you want to get the decision for
traffic_situation, decision = get_decision_for_row(row_index)
print(f"Traffic Situation for row {row_index}: {traffic_situation}")
print(f"Decision for row {row_index}: {decision}")
