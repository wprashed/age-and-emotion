import cv2
import numpy as np
import dlib

# Load pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
age_net = cv2.dnn.readNetFromCaffe("deploy_age.prototxt", "age_net.caffemodel")

# Define emotion labels and age ranges
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
AGE_LABELS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
AGE_RANGES = [(0, 2), (4, 6), (8, 12), (15, 20), (25, 32), (38, 43), (48, 53), (60, 100)]

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

    # Emotion detection rules
    if mouth_ratio > 0.35 and avg_eyebrow_height < 0.1:
        return "happy"
    elif avg_eyebrow_height > 0.18 and mouth_ratio < 0.05:
        return "surprise"
    elif avg_eyebrow_height < 0.06 and mouth_ratio < 0.05:
        return "sad"
    elif avg_eyebrow_height > 0.12 and avg_eye_height < 0.05:
        return "angry"
    elif avg_eye_height < 0.02:
        return "sleepy"
    elif mouth_ratio < 0.1 and avg_eyebrow_height < 0.08:
        return "neutral"
    else:
        return "unknown"

def estimate_age(face):
    """
    Estimate age using the pre-trained age model.
    """
    blob = cv2.dnn.blobFromImage(face, scalefactor=1.0, size=(227, 227), 
                                 mean=(78.4263377603, 87.7689143744, 114.895847746), 
                                 swapRB=False)
    age_net.setInput(blob)
    age_preds = age_net.forward().flatten()

    AGE_MIDPOINTS = [1, 5, 10, 17, 28, 40, 50, 80]
    corrected_preds = age_preds.copy()
    corrected_preds[4:] *= 2.5  # Increase weights for older age ranges

    corrected_preds /= np.sum(corrected_preds)

    face_height, face_width = face.shape[:2]
    if face_height > 100 or face_width > 100:
        corrected_preds[:4] = 0

    corrected_preds /= np.sum(corrected_preds)

    weighted_sum = sum(pred * mid for pred, mid in zip(corrected_preds, AGE_MIDPOINTS))
    estimated_age = int(weighted_sum)

    closest_range = min(AGE_RANGES, key=lambda x: abs((x[0] + x[1]) / 2 - estimated_age))
    age_index = AGE_RANGES.index(closest_range)

    return f"{AGE_LABELS[age_index]} (Est: {estimated_age})"

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera opened successfully. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame. Retrying...")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            try:
                face_roi = frame[y:y+h, x:x+w]
                rect = dlib.rectangle(x, y, x+w, y+h)
                landmarks = predictor(gray, rect)
                landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

                emotion = detect_emotion(landmarks, w, h)
                age = estimate_age(face_roi)

                cv2.putText(frame, f"Emotion: {emotion}", (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Age: {age}", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error processing face: {e}")
                continue

        cv2.imshow('Face Emotion & Age Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()