from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO('yolov8n.pt')

results = model(source = "D:\School\Fourth Year ENG\SE 4450 Software Engineering Design\Capstone\Images\Highway Test Image.jpeg", show = True, conf = 0.4, save = True) # Generator of results object