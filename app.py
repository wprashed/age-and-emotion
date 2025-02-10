import cv2
import numpy as np
import dlib

# Load pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
age_net = cv2.dnn.readNetFromCaffe("deploy_age.prototxt", "age_net.caffemodel")

# Define emotion labels
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Update the AGE_LABELS and add AGE_RANGES
AGE_LABELS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
AGE_RANGES = [(0, 2), (4, 6), (8, 12), (15, 20), (25, 32), (38, 43), (48, 53), (60, 100)]

def detect_emotion(landmarks, face_width, face_height):
    """
    Improved rule-based emotion detection using facial landmarks.
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

    # Add better rules for emotion detection
    if mouth_ratio > 0.35:  # Wide-open mouth
        return "happy"
    elif avg_eyebrow_height > 0.18 and mouth_ratio < 0.05:  # Raised eyebrows
        return "surprise"
    elif avg_eyebrow_height < 0.06 and mouth_ratio < 0.05:  # Drooping eyebrows and closed mouth
        return "sad"
    elif avg_eyebrow_height > 0.12 and avg_eye_height < 0.05:  # Lowered eyebrows and squinting eyes
        return "angry"
    elif avg_eye_height < 0.02:  # Closed eyes
        return "sleepy"
    elif mouth_ratio < 0.1 and avg_eyebrow_height < 0.08:  # Neutral expression
        return "neutral"
    else:
        return "unknown"

def estimate_age(face):
    """
    Estimate age using the pre-trained age model with bias correction and filtering.
    """
    blob = cv2.dnn.blobFromImage(face, scalefactor=1.0, size=(227, 227), 
                                 mean=(78.4263377603, 87.7689143744, 114.895847746), 
                                 swapRB=False)
    age_net.setInput(blob)
    age_preds = age_net.forward().flatten()

    # Print raw predictions
    print(f"Raw Age Predictions: {age_preds}")

    # Define the midpoint of each age range
    AGE_MIDPOINTS = [1, 5, 10, 17, 28, 40, 50, 80]

    # Apply stronger bias correction for adult ages
    corrected_preds = age_preds.copy()
    corrected_preds[4:] *= 2.0  # Increase weights for older age ranges

    # Normalize the corrected predictions
    corrected_preds /= np.sum(corrected_preds)

    # Filter out unrealistic predictions for adult faces
    face_height, face_width = face.shape[:2]
    if face_height > 100 or face_width > 100:  # Larger faces are likely adults
        corrected_preds[:4] = 0  # Ignore predictions for very young age ranges

    # Normalize again after filtering
    corrected_preds /= np.sum(corrected_preds)

    # Calculate the weighted average age
    total_pred = np.sum(corrected_preds)
    weighted_sum = sum(pred * mid for pred, mid in zip(corrected_preds, AGE_MIDPOINTS))
    estimated_age = int(weighted_sum / total_pred)

    # Print intermediate values
    print(f"Corrected Predictions: {corrected_preds}")
    print(f"Weighted Sum: {weighted_sum}, Total Pred: {total_pred}, Estimated Age: {estimated_age}")

    # Find the closest age range
    closest_range = min(AGE_RANGES, key=lambda x: abs((x[0] + x[1]) / 2 - estimated_age))
    age_index = AGE_RANGES.index(closest_range)

    return f"{AGE_LABELS[age_index]} (Est: {estimated_age})"

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
            emotion = "unknown"
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