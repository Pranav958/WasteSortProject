# ♻️ Waste Detection and Classification System

A computer vision–based system that detects objects in images and classifies waste into **biodegradable**, **non-biodegradable**, or **not waste** using a hybrid deep learning pipeline.

---

## 📌 Project Description

This project implements an intelligent waste detection system by combining "object detection" and "image classification" models. The system first detects objects using YOLOv8 and then classifies each detected object using a CNN-based classifier trained with transfer learning.



System Architecture:

1. YOLOv8 (General Object Detector)  
2. YOLOv8 (Waste-Specific Detector)  
3. CNN Classifier (MobileNetV2 – Transfer Learning)  
4. Post-processing (IOU filtering, duplicate removal)



Project Structure:

waste-detection-system/
│
├── integrated_pipeline.py
├── model1_detection.py
├── train_classifier.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
├── datasets/
├── outputs/


Installation:


pip install -r requirements.txt

How to Run:

python integrated_pipeline.py path/to/image.jpg


Model Details:

Detection: YOLOv8 (Ultralytics)

Classification: MobileNetV2 (Transfer Learning)

Classifier Accuracy: ~97%


Limitations:

Overall performance depends on object detection quality

Accuracy varies on complex or cluttered images


Future Improvements:

Real-time video support

Larger and more diverse dataset

Web or mobile deployment


Technologies Used:

Python

TensorFlow / Keras

YOLOv8

OpenCV

NumPy

Pillow(Python Imaging Library fork)