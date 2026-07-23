# Brain Tumor Detection using Deep Learning

## Overview

This project is a Brain Tumor Detection web application built using Deep Learning and Streamlit. It predicts whether an MRI brain scan contains a tumor and displays the prediction with confidence. The application provides a simple and user-friendly interface for uploading MRI images and viewing the results instantly.

## Features

- Upload MRI brain scan images
- Detect presence of brain tumor
- Display prediction confidence
- Clean and responsive Streamlit interface
- Fast image processing
- Easy to use

## Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- OpenCV
- NumPy
- Pillow
- Matplotlib

## Project Structure

```
Brain-Tumor-Detection/
│
├── app.py
├── brain_tumor_model.h5
├── requirements.txt
├── README.md
├── assets/
│   ├── images/
│   └── styles.css
├── pages/
├── utils/
└── dataset/
```

## Dataset

The model is trained on MRI brain scan images containing:

- Tumor
- No Tumor

Dataset images are preprocessed before training to improve model performance.

## Model

The project uses a Convolutional Neural Network (CNN) for image classification.

### Training Details

- Loss Function: Binary Crossentropy
- Optimizer: Adam
- Epochs: 10
- Image Size: 224 × 224
- Framework: TensorFlow / Keras

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Brain-Tumor-Detection.git
```

Move into the project directory

```bash
cd Brain-Tumor-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

## How to Use

1. Launch the Streamlit application.
2. Upload an MRI brain image.
3. Wait for the prediction.
4. View the detected result and confidence score.

## Future Improvements

- Multi-class tumor classification
- Tumor localization using Grad-CAM
- Explainable AI visualizations
- Patient history integration
- Cloud deployment
- Mobile-friendly interface

## Screenshots

Add screenshots of:

- Home Page
- Image Upload
- Prediction Result
- Confidence Score

## Author

**Udai Pratap Singh**

B.Tech CSE (AI)

Babu Banarasi Das University, Lucknow

## License

This project is licensed under the MIT License.