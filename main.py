import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "models/english_cnn.h5"
IMAGE_SIZE = 28
SCREEN_W, SCREEN_H = 640, 480

# ===============================
# LOAD MODEL
# ===============================
try:
    model = load_model(MODEL_PATH)
    MODEL_LOADED = True
    print("✅ Model loaded")
except:
    print("⚠️ Model not found. Running draw-only mode.")
    MODEL_LOADED = False

# ===============================
# MEDIAPIPE SETUP
# ===============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ===============================
# CANVAS & STATE
# ===============================
canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
points = []
text_output = ""

# ===============================
# INSTRUCTIONS
# ===============================
INSTRUCTIONS = [
    "👆 Use index finger to write",
    "✌️ 3 fingers → Space",
    "✊ Fist → Delete",
    "ESC to exit"
]

# ===============================
# COLOR PICKER (ON-SCREEN)
# ===============================
def nothing(x): pass

cv2.namedWindow("Color Picker")
cv2.resizeWindow("Color Picker", 300, 180)

cv2.createTrackbar("R", "Color Picker", 255, 255, nothing)
cv2.createTrackbar("G", "Color Picker", 0, 255, nothing)
cv2.createTrackbar("B", "Color Picker", 255, 255, nothing)

# ===============================
# FINGER STATE
# ===============================
def fingers_up(hand):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    fingers.append(hand.landmark[tips[0]].x < hand.landmark[tips[0]-1].x)

    # Other fingers
    for i in range(1, 5):
        fingers.append(hand.landmark[tips[i]].y < hand.landmark[tips[i]-2].y)

    return fingers

# ===============================
# CHARACTER PREDICTION
# ===============================
def predict_character(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
    gray = gray / 255.0
    gray = gray.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)

    pred = model.predict(gray, verbose=0)
    return chr(np.argmax(pred) + 65)

# ===============================
# MAIN LOOP
# ===============================
cap = cv2.VideoCapture(0)
cap.set(3, SCREEN_W)
cap.set(4, SCREEN_H)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # 🎨 Read color
    r = cv2.getTrackbarPos("R", "Color Picker")
    g = cv2.getTrackbarPos("G", "Color Picker")
    b = cv2.getTrackbarPos("B", "Color Picker")
    DRAW_COLOR = (b, g, r)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = None

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        finger_state = fingers_up(hand)
        index_up, middle_up = finger_state[1], finger_state[2]
        all_down = not any(finger_state)

        ix = int(hand.landmark[8].x * SCREEN_W)
        iy = int(hand.landmark[8].y * SCREEN_H)

        # ✍ WRITE
        if index_up and not middle_up:
            points.append((ix, iy))
            cv2.circle(frame, (ix, iy), 6, DRAW_COLOR, -1)

        # ␣ SPACE
        elif index_up and middle_up:
            gesture = "SPACE"

        # ⌫ DELETE
        elif all_down:
            gesture = "DELETE"

    # ===============================
    # DRAW STROKES
    # ===============================
    for i in range(1, len(points)):
        cv2.line(canvas, points[i-1], points[i], DRAW_COLOR, 5)

    # ===============================
    # GESTURE ACTIONS
    # ===============================
    if gesture == "SPACE":
        text_output += " "
        points.clear()

    elif gesture == "DELETE":
        text_output = text_output[:-1]
        points.clear()
        canvas[:] = 0

    # ===============================
    # CHARACTER PREDICTION
    # ===============================
    if len(points) > 35 and not result.multi_hand_landmarks:
        if MODEL_LOADED:
            char = predict_character(canvas)
            text_output += char
        points.clear()
        canvas[:] = 0

    # ===============================
    # MERGE CANVAS WITH FRAME
    # ===============================
    mask = canvas.sum(axis=2) > 0
    frame[mask] = canvas[mask]

    # ===============================
    # UI
    # ===============================
    # Draw semi-transparent overlay for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (SCREEN_W, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw text output
    cv2.putText(frame, f"Text: {text_output}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw instructions
    for i, instruction in enumerate(INSTRUCTIONS):
        y_pos = SCREEN_H - 40 - (len(INSTRUCTIONS) - i - 1) * 30
        cv2.putText(
            frame,
            instruction,
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
    # Draw current color indicator
    cv2.circle(frame, (SCREEN_W - 30, 30), 15, DRAW_COLOR, -1)
    cv2.circle(frame, (SCREEN_W - 30, 30), 16, (255, 255, 255), 1)

    cv2.imshow("✍ AI Air Writing", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
