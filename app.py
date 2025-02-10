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

# Define the midpoint of each age range
AGE_MIDPOINTS = [
    1, 5, 7, 10, 13, 16, 19, 22, 25, 28, 
    31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 
    61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 95
]

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
    mouth_height = abs(mouth[3][1] - mouth[9][1])  # Vertical distance of the mouth
    left_eyebrow_height = abs(left_eyebrow[0][1] - left_eyebrow[-1][1])
    right_eyebrow_height = abs(right_eyebrow[0][1] - right_eyebrow[-1][1])
    eyebrow_distance = abs(left_eyebrow[-1][0] - right_eyebrow[0][0])  # Horizontal distance between eyebrows
    mouth_corner_left = mouth[0][1]  # Left corner of the mouth
    mouth_corner_right = mouth[6][1]  # Right corner of the mouth

    print(f"Mouth Height: {mouth_height}, Eyebrow Distance: {eyebrow_distance}, Mouth Corners: {mouth_corner_left}, {mouth_corner_right}")

    # Rules for emotion detection
    if mouth_height > 25:  # Mouth open wide
        return "Happy"
    elif left_eyebrow_height > 20 or right_eyebrow_height > 20:  # Eyebrows raised
        return "Surprise"
    elif eyebrow_distance < 40:  # Eyebrows closer together
        return "Angry"
    elif mouth_corner_left > mouth[9][1] + 10 and mouth_corner_right > mouth[9][1] + 10:  # Mouth corners drooping
        return "Sad"
    else:
        return "Neutral"

def estimate_age(face):
    """
    Estimate age using the pre-trained age model.
    """
    blob = cv2.dnn.blobFromImage(face, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    age_net.setInput(blob)
    age_preds = age_net.forward()

    # Calculate the weighted average age
    weighted_age = sum(prob * mid for prob, mid in zip(age_preds[0], AGE_MIDPOINTS))
    total_prob = sum(age_preds[0])
    estimated_age = int(weighted_age / total_prob)

    return str(estimated_age)

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
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

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
            emotion = detect_emotion(landmarks)
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