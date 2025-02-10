# Face Emotion & Age Detection System

## Overview
This application detects faces in real-time using a webcam, recognizes known faces, estimates emotions and ages, and generates detailed reports (CSV and PDF) based on the detection logs. It uses advanced libraries like `face_recognition`, `dlib`, and `OpenCV` for face detection, emotion recognition, and age estimation.

---

## Features
1. **Real-Time Face Detection**:
   - Detects faces using Haar Cascade and `face_recognition`.
   - Draws bounding boxes around detected faces.

2. **Emotion Recognition**:
   - Recognizes emotions such as happy, sad, angry, neutral, etc., using facial landmarks.

3. **Age Estimation**:
   - Estimates the age of detected faces using a pre-trained Caffe model.

4. **Known Face Recognition**:
   - Recognizes known faces from the `known_faces` directory.
   - Automatically saves new faces and prompts for the user's name.

5. **Log Generation**:
   - Logs all detections (name, emotion, age) into a file (`logs/detection_log.txt`).

6. **Report Generation**:
   - Generates a **CSV report** summarizing emotions and ages for each user.
   - Generates a **PDF report** with a clean and professional design.

7. **Error Handling**:
   - Handles cases where face encoding fails or log files are missing.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- A webcam for real-time detection

### Dependencies
Install the required libraries using the following command:

```bash
pip install opencv-python dlib face_recognition pandas fpdf
```

### Pre-trained Models
Download the following pre-trained models and place them in the project directory:
1. **Shape Predictor Model**: [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
2. **Age Estimation Model**:
   - `deploy_age.prototxt`
   - `age_net.caffemodel`

Place these files in the root directory of the project.

---

## Directory Structure
The project has the following structure:
```
project/
│
├── known_faces/       # Directory to store known face images
│   ├── Alice/
│   │   ├── Alice_20231005_143005.jpg
│   ├── Bob/
│   │   ├── Bob_20231005_143200.jpg
│
├── logs/              # Directory to store logs and reports
│   ├── detection_log.txt
│   ├── report.csv
│   ├── report.pdf
│
├── shape_predictor_68_face_landmarks.dat
├── deploy_age.prototxt
├── age_net.caffemodel
├── app.py             # Main script
└── README.md          # This file
```

---

## Usage

### Running the Application
1. Start the application by running the following command:
   ```bash
   python app.py
   ```

2. The webcam will open, and the system will start detecting faces in real-time.

3. **Key Commands**:
   - Press `q` to quit the application.
   - Press `r` to generate a CSV and PDF report.

### Adding Known Faces
- Place images of known faces in the `known_faces` directory under subfolders named after the person (e.g., `known_faces/Alice/`).
- Restart the application to load the new faces.

---

## Report Generation
When you press `r`, the system generates two reports:
1. **CSV Report**:
   - File: `logs/report.csv`
   - Contains details about detected emotions, counts, average age, and all recorded ages for each user.

2. **PDF Report**:
   - File: `logs/report.pdf`
   - A visually appealing summary of emotions and ages for each user.

---

## Example Log Entry
The log file (`logs/detection_log.txt`) contains entries like this:
```
2023-10-05 14:30:00 - Name: Alice, Emotion: happy, Age: (25-32) (Est: 28)
2023-10-05 14:30:01 - Name: Bob, Emotion: angry, Age: (38-43) (Est: 40)
```

---

## Troubleshooting

### 1. Missing Log File
If the log file (`logs/detection_log.txt`) is missing:
- Ensure the `logs` directory exists. If not, create it manually:
  ```bash
  mkdir logs
  ```

### 2. No Faces Detected
- Ensure proper lighting and that the face is within the camera's view.
- Adjust the `scaleFactor` and `minNeighbors` parameters in the `detectMultiScale` function if necessary.

### 3. Errors During Installation
- If you encounter issues installing `dlib`, ensure you have the required build tools installed:
  - On Windows: Install [CMake](https://cmake.org/) and Visual Studio Build Tools.
  - On Linux: Install `cmake` and `build-essential`.

---

## Contributing
Contributions are welcome! If you find any bugs or want to add new features, feel free to submit a pull request.