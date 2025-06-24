# Vehicle License Plate Detection Using Deep Learning

This project aims to develop a system for detecting and recognizing vehicle license plates using deep learning techniques. The system identifies the location of the license plate, extracts its text characters, and identifies the vehicle's region of origin based on the license plate prefix.

## Key Features

* **Automatic License Plate Detection:** Employs a Faster R-CNN model to accurately localize license plate bounding boxes within images.
* **Character Recognition:** Utilizes the EasyOCR library to extract alphanumeric text from the detected license plates.
* **Region of Origin Identification:** Matches the extracted license plate prefixes with a database of Indonesian vehicle regions of origin.
* **Interactive Web Interface:** Features a web application built with Gradio for an easy-to-use system demonstration.

## Model Architecture

This system integrates two main deep learning components:

1.  **Object Detection:**
    * **Faster R-CNN** is used as the primary object detection architecture. This model was chosen for its balance of accuracy and efficiency in object detection.
    * [Optional: Mention the backbone used, e.g., "with a ResNet-50 FPN backbone."]

2.  **Optical Character Recognition (OCR):**
    * The **EasyOCR** library is integrated to perform text recognition from the license plate areas detected by Faster R-CNN.

## Dataset

This project uses an Indonesian vehicle license plate dataset consisting of **[Number of Images, e.g., 1000]** images. Each image has been annotated with precise bounding boxes for the license plate locations and corresponding character labels. The data was obtained from **[Mention source if applicable, e.g., a custom-collected dataset / Roboflow dataset]**.

## Model Training Process

The Faster R-CNN model was trained for **[Number of Epochs, e.g., 50]** epochs using **[Mention hardware, e.g., NVIDIA Tesla T4 GPU on Google Colab Pro]**. The training process focused on optimizing **Mean Average Precision (mAP)** as the primary object detection metric, along with Precision and Recall. [Optional: Add details such as *optimizer*, *learning rate*, or *loss function* if relevant.]

## Pre-trained Model

The trained Faster R-CNN model (`frcnn.pth`) has a large file size and cannot be directly uploaded to GitHub. You can download it via the following Google Drive link:

[**Download frcnn.pth Model from Google Drive**](https://drive.google.com/file/d/1tzRnGFFSHqf5i2Dd03rY0Q1r4LS6HuXj/view?usp=sharing)
*(Ensure this link is publicly accessible)*

## Development Environment (Requirements)

To run this project locally, ensure you have a Python environment with the following dependencies. You can install them using `pip`:

```bash
pip install -r requirements.txt
