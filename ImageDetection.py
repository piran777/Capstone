import torch
from ultralytics import YOLO
import csv
import pandas as pd

# Load a pretrained YOLO model
model = YOLO('yolov8n.pt')

file_path = './ImagePaths.csv'
custom_encoding = 'ISO-8859-1'

df = pd.read_csv(file_path, encoding = custom_encoding)

paths_data_array = df['path'].to_numpy()
print(paths_data_array) 

results = model(source = './Images/Highway Test Image.jpeg', conf = 0.4, save = False) # Generator of results object

carCount = 0
bikeCount = 0
busCount = 0
TruckCount = 0
none = 0

for box in results[0].boxes:
        class_id = int(box.cls)  # Get class ID
        class_label = results[0].names[class_id]  # Get class label from class ID

        match class_label:
                case "car":
                        carCount += 1
                case "truck":
                        TruckCount += 1
                case "bus":
                        busCount += 1
                case "bike":
                        bikeCount += 1
                case _:
                        none += 1

total = carCount + bikeCount + busCount + TruckCount

header = ['Time','Date', 'Day of the week', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total']
data = ['3:30:00 PM', '6', 'Wednesday', carCount, bikeCount, busCount, TruckCount, total]

with open('TrafficImageDetected.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(data)