# Face Emotion and Age Detection

![Face Emotion and Age Detection]

This project uses computer vision techniques to detect emotions and estimate the age of individuals in real-time video streams. It combines rule-based emotion detection using facial landmarks with a pre-trained deep learning model for age estimation.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Troubleshooting](#troubleshooting)
6. [Acknowledgments](#acknowledgments)

---

## Project Overview

This application detects emotions (e.g., Happy, Sad, Angry, etc.) and estimates the age of faces captured by a webcam. It uses:
- **OpenCV**: For face detection and video capture.
- **dlib**: For facial landmark detection.
- **Pre-trained Age Model**: A Caffe-based model for age estimation.

The emotion detection is rule-based, leveraging geometric relationships between facial landmarks, while the age estimation uses a weighted average approach to refine predictions.

---

## Features

- **Real-Time Emotion Detection**:
  - Detects emotions such as "Happy," "Sad," "Angry," "Surprise," "Neutral," and "Sleepy."
  - Uses facial landmarks to infer emotions based on geometric relationships.

- **Age Estimation**:
  - Estimates age ranges (e.g., `(25-32)`) and provides a single estimated age value (e.g., `28`).
  - Includes bias correction to improve accuracy for adult faces.

- **Lightweight and No Deep Learning for Emotions**:
  - Avoids TensorFlow or PyTorch for emotion detection, relying on simple rule-based logic.

- **Customizable**:
  - Easily adjust thresholds for emotion detection and age estimation.

---

## Installation

### Prerequisites
- Python 3.7 or higher
- OpenCV (`cv2`)
- dlib
- NumPy

### Steps to Install Dependencies

1. Clone the repository:
   ```bash
   git clone https://github.com/wprashed/age-and-emotion
   cd face-emotion-age-detection
   ```

2. Install required libraries:
   ```bash
   pip install opencv-python opencv-contrib-python dlib numpy
   ```

3. Download Pre-trained Models:
   - **Facial Landmark Predictor**:
     - Download `shape_predictor_68_face_landmarks.dat` from [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
     - Extract and place it in the project directory.
   - **Age Estimation Model**:
     - Download `deploy_age.prototxt` and `age_net.caffemodel` from [here](https://github.com/spmallick/learnopencv/tree/master/AgeGender/CaffeModels).
     - Place these files in the project directory.

---

## Usage

1. Run the script:
   ```bash
   python app.py
   ```

2. The webcam feed will open, displaying:
   - A rectangle around each detected face.
   - The detected emotion (e.g., "Happy") above the face.
   - The estimated age (e.g., `(25-32) (Est: 28)`) below the emotion.

3. Press `q` to quit the application.

---

## Troubleshooting

### Common Issues and Fixes

1. **Camera Not Opening**:
   - Ensure your webcam is connected and accessible.
   - Check permissions in **System Preferences > Privacy & Security > Camera**.

2. **Error Loading dlib Shape Predictor**:
   - Verify that `shape_predictor_68_face_landmarks.dat` is downloaded and placed in the correct directory.

3. **Incorrect Age Estimates**:
   - Adjust the bias correction multiplier in the `estimate_age` function.
   - Ensure proper lighting and camera quality for better results.

4. **Incorrect Emotions**:
   - Fine-tune the thresholds in the `detect_emotion` function.
   - Test with static images to verify the logic.

---

## Acknowledgments

- **OpenCV**: For providing tools for face detection and video processing.
- **dlib**: For facial landmark detection.
- **Caffe Age Model**: For age estimation, sourced from [LearnOpenCV](https://github.com/spmallick/learnopencv).
