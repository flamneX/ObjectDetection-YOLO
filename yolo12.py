"""
YOLO12.py

Original file is located at
    https://colab.research.google.com/drive/1o8dOpDDcQiXBUwQ5TiBHcUInBi_Zi5Dh?usp=sharing
"""

# Install necessary dependencies (run this in your terminal before running the script)
# pip install ultralytics
# pip install roboflow

# Import required libraries
import ultralytics
from ultralytics import YOLO
from roboflow import Roboflow

# Perform the Ultraytics checks
ultralytics.checks()

# Roboflow API setup
rf = Roboflow(api_key="BuDn3mPWk1426lqrnQRc")
project = rf.workspace("test-vivys").project("tumor-detection-ko5jp-4h325")
version = project.version(1)
dataset = version.download("yolov12")

# Load the YOLOv12 model
model = YOLO("yolo12n.pt")

# Train the model
model.train(
    data=f"{dataset.location}/data.yaml",  # Dataset path
    epochs=50,
    batch=16,
    imgsz=640,
    workers=8,
    optimizer="auto",
    lr0=0.001,
    lrf=0.00001,
    momentum=0.937,
    box=0.2,
    degrees=0.0,
    translate=0.1,
    shear=0.0,
    perspective=0.0,
    resume=False,
    project="yolo12_original",
    name="exp"
)

# Save path to original weights
original_12_path = "yolo12_original/exp/weights/best.pt"

# Train the fine-tuned model
model.train(
    data=f"{dataset.location}/data.yaml",  # Dataset path
    epochs=50,
    batch=16,
    imgsz=640,
    workers=8,
    optimizer="auto",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    box=0.2,
    degrees=0.0,
    translate=0.1,
    shear=0.0,
    perspective=0.0,
    resume=False,
    project="yolo12_finetune",
    name="exp"
)

# Save path to fine-tuned weights
finetuned_12_path = "yolo12_finetune/exp/weights/best.pt"

# === Load original and fine-tuned models ===
original_12_model = YOLO(original_12_path)
fine_tuned_12_model = YOLO(finetuned_12_path)

# === TRAINING SET ===
results_original_12_train = original_12_model.val(
    data=f"{dataset.location}/data.yaml", 
    split='train', 
    save_conf=True, 
    plots=True, 
    name="original_12_train"
)

results_finetuned_12_train = fine_tuned_12_model.val(
    data=f"{dataset.location}/data.yaml", 
    split='train', 
    save_conf=True, 
    plots=True, 
    name="finetuned_12_train"
)

# === VALIDATION SET ===
results_original_12_val = original_12_model.val(
    data=f"{dataset.location}/data.yaml", 
    split='val', 
    save_conf=True, 
    plots=True, 
    name="original_12_val"
)

results_finetuned_12_val = fine_tuned_12_model.val(
    data=f"{dataset.location}/data.yaml", 
    split='val', 
    save_conf=True, 
    plots=True, 
    name="finetuned_12_val"
)

# === TESTING SET ===
results_original_12_test = original_12_model.val(
    data=f"{dataset.location}/data.yaml", 
    split='test', 
    save_conf=True, 
    plots=True, 
    name="original_12_test"
)

results_finetuned_12_test = fine_tuned_12_model.val(
    data=f"{dataset.location}/data.yaml", 
    split='test', 
    save_conf=True, 
    plots=True, 
    name="finetuned_12_test"
)
