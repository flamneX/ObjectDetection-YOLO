"""YOLO11.py

Original file is located at
    https://colab.research.google.com/drive/1Docu4WhE7yqAVa_PxzLJTIwq9MnDv5gc?usp=sharing
"""

# Install necessary dependencies (run this in your terminal before running the script)
# pip install ultralytics
# pip install roboflow

# Import required libraries
import ultralytics
from ultralytics import YOLO
from roboflow import Roboflow
import shutil
import os

# Perform the Ultraytics checks
ultralytics.checks()

# Roboflow API setup
rf = Roboflow(api_key="BuDn3mPWk1426lqrnQRc")
project = rf.workspace("test-vivys").project("tumor-detection-ko5jp-4h325")
version = project.version(1)
dataset = version.download("yolov11")

# Load the YOLO11n model
model = YOLO("yolo11n.pt")

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
    project="yolo11_original",
    name="exp"
)

# Save path to original weights
original_11_path = "yolo11_original/exp/weights/best.pt"

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
    project="yolo11_finetune",
    name="exp"
)

# Save path to fine-tuned weights
finetuned_11_path = "yolo11_finetune/exp/weights/best.pt"

# === Load original and fine-tuned models ===
original_11_model = YOLO(original_11_path)
fine_tuned_11_model = YOLO(finetuned_11_path)

# === TRAINING SET ===
results_original_11_train = original_11_model.val(
    data=f"{dataset.location}/data.yaml", 
    split='train', 
    save_conf=True, 
    plots=True, 
    name='original_11_train'
)

results_finetuned_11_train = fine_tuned_11_model.val(
    data=f"{dataset.location}/data.yaml", 
    split='train', 
    save_conf=True, 
    plots=True, 
    name='finetuned_11_train'
)

# === VALIDATION SET ===
results_original_11_val = original_11_model.val(
    data=f"{dataset.location}/data.yaml", 
    split='val', 
    save_conf=True, 
    plots=True, 
    name='original_11_val'
)

results_finetuned_11_val = fine_tuned_11_model.val(
    data=f"{dataset.location}/data.yaml", 
    split='val', 
    save_conf=True, 
    plots=True, 
    name='finetuned_11_val'
)

# === TESTING SET ===
results_original_11_test = original_11_model.val(
    data=f"{dataset.location}/data.yaml", 
    split='test', 
    save_conf=True, 
    plots=True, 
    name='original_11_test'
)

results_finetuned_11_test = fine_tuned_11_model.val(
    data=f"{dataset.location}/data.yaml", 
    split='test', 
    save_conf=True, 
    plots=True, 
    name='finetuned_11_test'
)