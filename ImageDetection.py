import torch
from ultralytics import YOLO
import csv
import os
import pandas as pd

# Load a pretrained YOLO model
model = YOLO('yolov8n.pt')

imagefolder_path = './Images'
imageNames = []

csvfile_path = './ImagePaths.csv'
custom_encoding = 'ISO-8859-1'

obj = os.scandir(imagefolder_path) #Scans images folder for all the image names

for entry in obj : #Gets image names
    if entry.is_dir() or entry.is_file():
        imageNames.append("./Images/" + entry.name)

with open('ImagePaths.csv', 'w', newline='') as f: #Used to write all image paths for all the images into a csv
    writer = csv.writer(f)
    writer.writerow(["path"])
    writer.writerows([[name] for name in imageNames])

obj.close()

print(imageNames)
df = pd.read_csv(csvfile_path, encoding=custom_encoding)

paths_data_array = df['path'].to_numpy()

data = []



for i in paths_data_array:
        results = model(source=i, conf=0.4, save=False)  # Generator of results object

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

        # Define a function to determine the traffic situation
        def determine_traffic_situation(total_count):
                if total_count >= 50:
                        return "heavy"
                elif 20 <= total_count < 50:
                        return "normal"
                else:
                        return "low"

        # Use the function to get the traffic situation
        traffic_situation = determine_traffic_situation(total)

        
        data.append(['3:30:00 PM', '6', 'Wednesday', carCount, bikeCount, busCount, TruckCount, total, traffic_situation])

header = ['Time', 'Date', 'Day of the week', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total', 'Traffic Situation']

with open('TrafficImageDetected.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)
