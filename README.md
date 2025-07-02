
# ObjectDetection-YOLO
Assignment for UECS3413 Digital Image Processing

Title: Intracranial Tumor Detection and Classification Models Comparison Using YOLO Algorithms
This repository presents a comparative deep learning study using different YOLO versions (YOLOv8, YOLOv11, and YOLOv12) for tumor detection in medical imagery.


## Authors
- [@Cammy276](https://github.com/Cammy276)
- [@Yu-2008] (https://github.com/Yu-2008)
- [@flamneX] (https://github.com/flamneX)
- [@LIOWKEHAN] (https://github.com/LIOWKEHAN)


## Structures 
<pre> 
.
├── models/                                 # Contains all trained YOLO model versions
│
│   ├── yolo_v8/                            # YOLOv8-specific outputs & structure
│   │   ├── yolov8n.pt                      # Original Loaded YOLOv8 model
│   │   ├── yolo8_finetune/                 # Code/checkpoints/logs and best weight for fine-tuned YOLOv8
│   │   ├── yolo8_original/                 # Code/checkpoints/logs and best weight for original YOLOv8
│   │   ├── .config/                        # Environment/Colab system config
│   │   ├── runs/
│   │   │   └── detect/
│   │   │       ├── original_8_train/       # Original model on training set
│   │   │       ├── original_8_val/         # Original model on validation set
│   │   │       ├── original_8_test/        # Original model on testing set
│   │   │       ├── finetuned_8_train/      # Fine-tuned model on training set
│   │   │       ├── finetuned_8_valid/      # Fine-tuned model on validation set
│   │   │       └── finetuned_8_test/       # Fine-tuned model on testing set
│   │   └── Tumor-Detection-1/
│   │       ├── data.yaml                   # Dataset config (classes, paths)
│   │       ├── README.roboflow.txt         # Roboflow export metadata
│   │       ├── train/
│   │       │   ├── images/
│   │       │   ├── labels/
│   │       │   └── labels.cache
│   │       ├── valid/
│   │       │   ├── images/
│   │       │   ├── labels/
│   │       │   └── labels.cache
│   │       └── test/
│   │           ├── images/
│   │           ├── labels/
│   │           └── labels.cache
│
│   ├── yolo_v11/
│   │   └── ... # (same structure)
│
│   └── yolo_v12/
│       └── ... # (same structure)
│
└── README.md                               # Main project overview
</pre>

## Tehcnologies Used
| Tool / Library                           | Description                                                                
| -----------------------------------------|-------------------------------------------------------------------------- 
| **Ultralytics YOLOv8, YOLO11n, YOLO12n** | Object detection framework for training, testing, and evaluation tasks.    
| **PyTorch**                              | Backend deep learning framework powering YOLO models.                    
| **Roboflow**                             | Dataset management, annotation, and export to YOLO format.                 
| **Python 3.x**                           | Main programming language for scripts and training pipelines.              
| **Google Colab**                         | Cloud-based notebook platform used for GPU-accelerated training & testing.


## Objectives

1) Train several YOLO algorithm versions (YOLOv8, YOLOv11, **YOLOv12) to detect brain tumors (glioma, meningioma, pituitary, space-occupying lesions)

2) Compare the performance matrixes between the YOLO models to find the best model
**The application of YOLOv12 in the detection and classification of intracranial tumors is currently limited


## Methodology
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


## Experiment Result

| Fine-tuned Model | Precision | Recall | F1 Score | mAP50 | mAP50–95 |
|------------------|-----------|--------|----------|--------|-----------|
| YOLOv8           | 0.694 | 0.600  | 0.644 | 0.635 | 0.508 |
| YOLOv11          | 0.650 | 0.643  | 0.646 | 0.653 | 0.502 |
| YOLOv12          | 0.572 | 0.757 | 0.652  | 0.635 | 0.498 |



## Challenges & Potential Improvements

### 1. Imbalanced Dataset
**Challenges:**
- Certain classes (e.g., glioma, space-occupying lesion) are underrepresented
- Low performance metrics
- Overfitting issues

**Potential Improvements:**
- Use class-weighted loss functions
- Use focal loss

---

### 2. Limited GPU Access
**Challenges:**
- Free Colab disconnects frequently
- Short GPU availability

**Potential Improvements:**
- Upgrade to Colab Pro
- Use free cloud credits from AWS or GCP
- Train during off-peak hours

---

### 3. Model Selection Difficulties
**Challenges:**
- Confusion choosing among YOLO variants (n, s, m, l)
- Not enough time/expertise to test all

**Potential Improvements:**
- Train each variant for a few epochs and compare
- Refer to YOLO documentation and benchmarks
- Use automated tuning tools


## Conclusion

| Model Version | Highest Performance Metric        | Applications                                                                 |
|---------------|-----------------------------------|------------------------------------------------------------------------------|
| YOLOv8        | Precision                         | High accuracy, low false positives; preliminary tumor screening or secondary confirmation tools |
| YOLOv11       | Balanced between precision & recall | Balanced, dependable results; general diagnostics                            |
| YOLOv12       | Recall                            | High detection rate, catches more cases; critical or high-sensitivity environments (missing a tumor could be life-threatening) |


## Contributing

Contributions are always welcome!

To get started:

1. **Fork** the repository to your GitHub account.
2. **Create a new branch** for your feature or fix:  
   `git checkout -b your-feature-name`
3. **Make your changes** and commit them with a clear message:  
   `git commit -m "Add: Description of your change"`
4. **Push** your branch to your forked repository:  
   `git push origin your-feature-name`
5. **Open a Pull Request** from your branch to the main project.

Feel free to open an issue first if you'd like to discuss your idea before implementing it.
