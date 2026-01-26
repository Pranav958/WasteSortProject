# ♻️ WasteSort: Hybrid Waste Detection & Classification System

## 📌 Overview
WasteSort is an end-to-end computer vision system that detects and classifies waste items from images.  
The project is designed to support automated waste segregation and recycling workflows by combining object detection and image classification models.

This system follows a **hybrid pipeline**:
- Object Detection to localize waste items
- Image Classification to identify the waste category

---

## 🚀 Features
- Detects waste objects in images using **YOLOv8**
- Classifies detected waste using a **MobileNetV2 CNN**
- End-to-end Python-based inference pipeline
- Modular design (detection + classification stages)
- Applicable to real-world waste management scenarios

---

## 🧠 Technical Approach

### 1. Object Detection
- Model: YOLOv8  
- Purpose: Detect waste objects and generate bounding boxes  

### 2. Waste Classification
- Model: MobileNetV2 (CNN)  
- Framework: TensorFlow / Keras  
- Purpose: Classify detected waste into predefined categories  

### 3. Pipeline Flow
1. Input image is provided to the system  
2. YOLOv8 detects waste regions  
3. Detected regions are cropped  
4. Cropped images are passed to the classifier  
5. Final labeled output is generated  

---

## 🛠️ Tech Stack
- Programming Language: Python  
- Deep Learning: TensorFlow, Keras  
- Computer Vision: OpenCV  
- Object Detection: YOLOv8  
- Data Processing: NumPy, Pillow  

---

## 📊 Results
- Achieved approximately **97% classification accuracy** on validation/testing data  
- Successfully detects and classifies waste across multiple sample images  
- Demonstrates robustness across varied lighting conditions and backgrounds  

*Performance may vary depending on dataset quality and image conditions.*

---

## 📂 Project Structure
```
WasteSortProject/
│
├── yolo_detect.py          # YOLOv8-based waste detection
├── classifier.py           # CNN-based waste classification
├── train_classifier.py     # Training script for MobileNetV2
├── requirements.txt        # Project dependencies
├── README.md               # Documentation
└── sample_images/          # Example input images
```

---

## ▶️ How to Run

### 1. Clone the Repository
```
git clone https://github.com/Pranav958/WasteSortProject.git
cd WasteSortProject
```

### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Run the Detection & Classification Pipeline
```
python yolo_detect.py
```

---

## 💡 Applications
- Automated waste segregation systems  
- Smart recycling solutions  
- Environmental monitoring tools  
- Computer vision pipelines for object classification  

---

## 🔮 Future Improvements
- Extend support to real-time video streams  
- Increase number of waste categories  
- Deploy as a web or mobile application  
- Improve dataset diversity and scale  
