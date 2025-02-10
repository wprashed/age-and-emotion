import cv2

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load OpenCV's pre-trained facial landmark detector
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("lbfmodel.yaml")  # Download the LBF model from OpenCV's GitHub repository

def detect_emotion(landmarks):
    """
    Rule-based emotion detection using facial landmarks.
    """
    # Extract key points for mouth, eyebrows, and eyes
    mouth = landmarks[0][48:68]  # Mouth landmarks
    left_eye = landmarks[0][36:42]  # Left eye landmarks
    right_eye = landmarks[0][42:48]  # Right eye landmarks
    left_eyebrow = landmarks[0][17:22]  # Left eyebrow landmarks
    right_eyebrow = landmarks[0][22:27]  # Right eyebrow landmarks

    # Calculate distances between key points
    mouth_height = abs(mouth[3][1] - mouth[9][1])  # Vertical distance of the mouth
    left_eyebrow_height = abs(left_eyebrow[0][1] - left_eye[1][1])
    right_eyebrow_height = abs(right_eyebrow[-1][1] - right_eye[1][1])

    # Simple rules for emotion detection
    if mouth_height > 10:  # Mouth open wide
        return "Happy"
    elif left_eyebrow_height < 5 or right_eyebrow_height < 5:  # Eyebrows raised
        return "Surprise"
    else:
        return "Neutral"

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Detect facial landmarks
        _, landmarks = facemark.fit(gray, faces)
        if landmarks:
            emotion = detect_emotion(landmarks)
        else:
            emotion = "Unknown"

        # Draw rectangle and put text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"Emotion: {emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Face Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()