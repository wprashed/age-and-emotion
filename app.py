import cv2
import dlib

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load dlib's pre-trained facial landmark detector
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except Exception as e:
    print(f"Error loading dlib shape predictor: {e}")
    print("Please download the file from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 and extract it.")
    exit()

def detect_emotion(landmarks):
    """
    Rule-based emotion detection using facial landmarks.
    """
    # Extract key points for mouth, eyebrows, and eyes
    mouth = landmarks[48:68]  # Mouth landmarks
    left_eye = landmarks[36:42]  # Left eye landmarks
    right_eye = landmarks[42:48]  # Right eye landmarks
    left_eyebrow = landmarks[17:22]  # Left eyebrow landmarks
    right_eyebrow = landmarks[22:27]  # Right eyebrow landmarks

    # Calculate distances between key points
    mouth_height = abs(mouth[3].y - mouth[9].y)  # Vertical distance of the mouth
    left_eyebrow_height = abs(left_eyebrow[0].y - left_eye[1].y)
    right_eyebrow_height = abs(right_eyebrow[-1].y - right_eye[1].y)

    # Simple rules for emotion detection
    if mouth_height > 10:  # Mouth open wide
        return "Happy"
    elif left_eyebrow_height < 5 or right_eyebrow_height < 5:  # Eyebrows raised
        return "Surprise"
    else:
        return "Neutral"

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
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Detect facial landmarks using dlib
        rect = dlib.rectangle(x, y, x+w, y+h)
        try:
            landmarks = predictor(gray, rect)
            landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
            emotion = detect_emotion(landmarks)
        except Exception as e:
            emotion = "Unknown"
            print(f"Error detecting landmarks: {e}")

        # Display the detected emotion
        cv2.putText(frame, f"Emotion: {emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Emotion Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()