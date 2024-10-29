import torch 
import numpy

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5m")  # or yolov5n - yolov5x6, custom

# Images
img = "C:\\Users\\richard\\Downloads\\test_2.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.show()  # or .show(), .save(), .crop(), .pandas(), etc.import torch
