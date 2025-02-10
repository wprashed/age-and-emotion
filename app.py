import cv2
import dlib
import numpy as np

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load dlib's pre-trained facial landmark detector
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except Exception as e:
    print(f"Error loading dlib shape predictor: {e}")
    print("Please download the file from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 and extract it.")
    exit()

# Load pre-trained age model
age_net = cv2.dnn.readNetFromCaffe("deploy_age.prototxt", "age_net.caffemodel")

# Define refined age labels
AGE_LABELS = [
    '(0-2)', '(3-5)', '(6-8)', '(9-11)', '(12-14)', 
    '(15-17)', '(18-20)', '(21-23)', '(24-26)', '(27-29)', 
    '(30-32)', '(33-35)', '(36-38)', '(39-41)', '(42-44)', 
    '(45-47)', '(48-50)', '(51-53)', '(54-56)', '(57-59)', 
    '(60-62)', '(63-65)', '(66-68)', '(69-71)', '(72-74)', 
    '(75-77)', '(78-80)', '(81-83)', '(84-86)', '(87-89)', 
    '(90-100)'
]

def detect_emotion(landmarks, face_width, face_height):
    """
    Rule-based emotion detection using facial landmarks.
    """
    # Extract key points for different facial features
    mouth = landmarks[48:68]  # Mouth landmarks
    left_eyebrow = landmarks[17:22]  # Left eyebrow landmarks
    right_eyebrow = landmarks[22:27]  # Right eyebrow landmarks
    left_eye = landmarks[36:42]  # Left eye landmarks
    right_eye = landmarks[42:48]  # Right eye landmarks

    # Normalize measurements based on face size
    normalize = lambda x: x / face_width

    # Calculate distances and ratios
    mouth_width = normalize(abs(mouth[6][0] - mouth[0][0]))
    mouth_height = normalize(abs(mouth[3][1] - mouth[9][1]))
    mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0

    left_eyebrow_height = normalize(abs(left_eyebrow[2][1] - left_eye[1][1]))
    right_eyebrow_height = normalize(abs(right_eyebrow[2][1] - right_eye[1][1]))
    avg_eyebrow_height = (left_eyebrow_height + right_eyebrow_height) / 2

    left_eye_height = normalize(abs(left_eye[1][1] - left_eye[5][1]))
    right_eye_height = normalize(abs(right_eye[1][1] - right_eye[5][1]))
    avg_eye_height = (left_eye_height + right_eye_height) / 2

    # Print debug information
    print(f"Mouth Ratio: {mouth_ratio:.3f}, Avg Eyebrow Height: {avg_eyebrow_height:.3f}, Avg Eye Height: {avg_eye_height:.3f}")

    # Simplified rules for emotion detection
    if mouth_ratio > 0.2:
        return "Happy"
    elif avg_eyebrow_height > 0.1:
        return "Surprised"
    elif mouth_ratio < 0.05 and avg_eyebrow_height < 0.05:
        return "Sad"
    elif avg_eyebrow_height > 0.08 and mouth_ratio < 0.1:
        return "Angry"
    elif avg_eye_height < 0.02:
        return "Sleepy"
    else:
        return "Neutral"

def estimate_age(face):
    """
    Estimate age using the pre-trained age model.
    """
    blob = cv2.dnn.blobFromImage(face, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    age_net.setInput(blob)
    age_preds = age_net.forward()

    # Find the age range with the highest probability
    age_index = age_preds[0].argmax()
    age_range = AGE_LABELS[age_index]

    return age_range

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera opened successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame. Retrying...")
        continue

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No faces detected.")
    else:
        print(f"{len(faces)} face(s) detected.")

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Detect facial landmarks using dlib
        try:
            rect = dlib.rectangle(x, y, x+w, y+h)
            landmarks = predictor(gray, rect)
            landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
            emotion = detect_emotion(landmarks, w, h)  # Pass face width and height
        except Exception as e:
            emotion = "Unknown"
            print(f"Error detecting landmarks: {e}")

        # Estimate age
        face_roi = frame[y:y+h, x:x+w]
        age = estimate_age(face_roi)

        # Display the detected emotion and age
        cv2.putText(frame, f"Emotion: {emotion}", (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Age: {age}", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Emotion & Age Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()