# ObjectDetection-YOLO
Assignment for UECS3413 Digital Image Processing

Title: Intracranial Tumor Detection and Classification Models Comparison Using YOLO Algorithms

Objectives: 
1) Train several YOLO algorithm versions (YOLOv8, YOLOv11, **YOLOv12) to detect brain tumors (glioma, meningioma, pituitary, space-occupying lesions)
2) Compare the performance matrixes between the YOLO models to find the best model
**The application of YOLOv12 in the detection and classification of intracranial tumors is currently limited

Methodology:
1) Data Collection
Total Images: 1986
- Train: 1370
- Valid: 395
- Test: 191

2) Preprocessing
- Auto Orientation
- Resize (640x640)
- Not null

3) Augmentation
- Horizontal flip (50%)
- Crops (0-20%)
- Shear (-10% - 10%)
- Noise (0.1%)

4) Model Training
- epochs: 50
- batch: 16
- imgsz: 640
- workers: 8
- optimizer: Auto
- lr0: 0.001
- lrf: 0.00001
- momentum: 0.937
- degrees: 0.0
- translate: 0.1
- shear: 0.0
- perspective: 0.0

5) Model Tuning
- epochs: 50
- batch: 16
- imgsz: 640
- workers: 8
- optimizer: Auto
- lr0: 0.01
- lrf: 0.01
- box: 0.2
- degrees: 0.0
- translate: 0.1
- shear: 0.0
- perspective: 0.0

Experimental Result:








