# 🚘 Vehicle License Plate Detection

A web-based vehicle license plate detection and recognition application using Faster R-CNN and EasyOCR, deployed with Gradio. Users can upload vehicle images and receive predictions for license plate location and extracted text, along with identification of the region of origin based on the plate prefix. The application also provides an interactive interface for easy demonstration.

## 🧠 Model Architecture
Faster R-CNN: This is the deep learning model specifically employed for vehicle license plate detection in images. The frcnn.pth file represents this trained Faster R-CNN model.
EasyOCR: This is a Python library that comes equipped with its own pre-trained models for optical character recognition (OCR). Its role is to extract text from the license plates that Faster R-CNN has detected.

## 🛠️ Tech Stack

Python (Core programming language)
Faster R-CNN (For object detection, specifically license plates)
EasyOCR (For Optical Character Recognition)
Gradio (For building the interactive web interface)
PyTorch (Underlying framework for Faster R-CNN, as frcnn.pth indicates a PyTorch model)
OpenCV (Likely used for image processing)
NumPy (For numerical operations)
Pillow (For image handling)
Hugging Face Spaces (For application deployment)

## 📊 Dataset

This project utilizes a custom-collected dataset sourced from Roboflow, under the project name "vehicle-and-license-plate".

Dataset Details:

Total Images: The dataset comprises 2,167 images.
Object Classes: The images are annotated with bounding boxes covering five object classes: bus, cars, license_plate, motorcycle, and truck. While the dataset contains these various vehicle types, the primary focus of this project is specifically on the license_plate object for further processing (OCR and region identification).
Annotations: Each image is meticulously annotated with bounding boxes around the vehicle license plate areas.
Image Conditions: The dataset was collected under diverse conditions and from various viewpoints, featuring varying resolutions. This diversity supports the model's generalization capabilities.

## ⚙️ Model Training Process

The Faster R-CNN model was trained for 20 epochs using SGD optimizer (with 0.9 momentum, 0.0005 weight decay, 0.01 initial learning rate) on Google Colaboratory's GPU. Performance was monitored using validation data, with mAP, Precision, and Recall as key metrics.

* **Model Training & Exploration Notebook:** [https://github.com/nurulkhazanah/Deteksi-plat-kendaraan/blob/main/Nuka_CarPlateDetection.ipynb](https://github.com/nurulkhazanah/Deteksi-plat-kendaraan/blob/main/Nuka_CarPlateDetection.ipynb)
    * `Nuka_CarPlateDetection.ipynb` (This file contains detailed steps for model training, data preprocessing, and initial explorations.)

## 📥 Pre-trained Model

The trained Faster R-CNN model (`frcnn.pth`) has a large file size and cannot be directly uploaded to GitHub. You can download it via the following Google Drive link:

[**Download frcnn.pth Model from Google Drive**](https://drive.google.com/file/d/1tzRnGFFSHqf5i2Dd03rY0Q1r4LS6HuXj/view?usp=sharing)

## 🚀 Application Deployment

The application is deployed as an interactive web demo using Gradio on Hugging Face Spaces.
[**License Plate Detection App on Hugging Face Spaces**] (https://huggingface.co/spaces/khalut/deteksi-plat-kendaraan)

## 👩🏼‍💻 How to Run the Project (Local)
To run this project on your local machine:

1.  **Clone this repository:**
    ```bash
    git clone [https://github.com/nurulkhazanah/Deteksi-plat-kendaraan.git](https://github.com/nurulkhazanah/Deteksi-plat-kendaraan.git)
    cd Deteksi-plat-kendaraan
    ```
2.  **Download the model:** Download the `frcnn.pth` file from the Google Drive link provided in the "Pre-trained Model" section above, then place it in the `Deteksi-plat-kendaraan` folder that you cloned.
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(The content of your `requirements.txt` file is as follows:)*
    ```
    gradio
    torch
    torchvision
    Pillow
    numpy
    easyocr
    ```
4.  **Run the application:**
    ```bash
    python app.py
    ```
    The application will run at `http://127.0.0.1:7860` (or another port indicated by Gradio in your console).

