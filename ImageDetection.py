import torch
from ultralytics import YOLO
import csv

# Load a pretrained YOLO model
model = YOLO('yolov8n.pt')

# Define a path to a CSV file with images
# source = '.\ImagePaths.csv'

results = model(source = './Images/Highway Test Image.jpeg', conf = 0.4, save = False) # Generator of results object

carCount = 0
bikeCount = 0
busCount = 0
TruckCount = 0
none = 0

for box in results[0].boxes:
        class_id = int(box.cls)  # Get class ID
        class_label = results[0].names[class_id]  # Get class label from class ID
        #print(f'Detected class: {class_label}')  # Print class label

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

