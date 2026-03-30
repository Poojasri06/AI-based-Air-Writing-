import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from PIL import Image
import io

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="AI Air Writing",
    page_icon="✍️",
    layout="wide"
)

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "models/english_cnn.h5"
IMAGE_SIZE = 28
MIN_POINTS_FOR_PREDICTION = 35  # Minimum points before character prediction

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_prediction_model():
    try:
        model = load_model(MODEL_PATH)
        return model, True
    except Exception as e:
        st.warning(f"⚠️ Model not found: {e}. Running in draw-only mode.")
        return None, False

model, MODEL_LOADED = load_prediction_model()

# ===============================
# MEDIAPIPE SETUP
# ===============================
@st.cache_resource
def get_mediapipe_hands():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    return mp_hands, hands

mp_hands, hands = get_mediapipe_hands()
mp_draw = mp.solutions.drawing_utils

# ===============================
# SESSION STATE
# ===============================
if 'canvas' not in st.session_state:
    st.session_state.canvas = np.zeros((480, 640, 3), dtype=np.uint8)
if 'points' not in st.session_state:
    st.session_state.points = []
if 'text_output' not in st.session_state:
    st.session_state.text_output = ""
if 'draw_color' not in st.session_state:
    st.session_state.draw_color = (255, 0, 255)

# ===============================
# HELPER FUNCTIONS
# ===============================
def fingers_up(hand):
    """
    Detect which fingers are up.
    Returns a list of booleans for each finger: [thumb, index, middle, ring, pinky]
    
    Note: Thumb uses left/right movement (x-axis) while other fingers use up/down (y-axis)
    """
    tips = [4, 8, 12, 16, 20]
    fingers = []
    
    # Thumb - check horizontal position (left/right)
    fingers.append(hand.landmark[tips[0]].x < hand.landmark[tips[0]-1].x)
    
    # Other fingers - check vertical position (up/down)
    for i in range(1, 5):
        fingers.append(hand.landmark[tips[i]].y < hand.landmark[tips[i]-2].y)
    
    return fingers

def predict_character(img, model):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
    gray = gray / 255.0
    gray = gray.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
    
    pred = model.predict(gray, verbose=0)
    return chr(np.argmax(pred) + 65)

def process_frame(frame, draw_color):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.flip(frame, 1)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    
    gesture = None
    
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        
        finger_state = fingers_up(hand)
        index_up, middle_up = finger_state[1], finger_state[2]
        all_down = not any(finger_state)
        
        h, w = frame.shape[:2]
        ix = int(hand.landmark[8].x * w)
        iy = int(hand.landmark[8].y * h)
        
        # ✍ WRITE
        if index_up and not middle_up:
            st.session_state.points.append((ix, iy))
            cv2.circle(frame, (ix, iy), 6, draw_color, -1)
            gesture = "WRITING"
        
        # ␣ SPACE
        elif index_up and middle_up:
            gesture = "SPACE"
        
        # ⌫ DELETE
        elif all_down:
            gesture = "DELETE"
    
    # Draw strokes on canvas
    canvas = st.session_state.canvas.copy()
    for i in range(1, len(st.session_state.points)):
        cv2.line(canvas, st.session_state.points[i-1], st.session_state.points[i], draw_color, 5)
    
    # Handle gestures
    if gesture == "SPACE":
        st.session_state.text_output += " "
        st.session_state.points.clear()
        st.session_state.canvas[:] = 0
    
    elif gesture == "DELETE":
        st.session_state.text_output = st.session_state.text_output[:-1]
        st.session_state.points.clear()
        st.session_state.canvas[:] = 0
    
    # Character prediction
    if len(st.session_state.points) > MIN_POINTS_FOR_PREDICTION and not result.multi_hand_landmarks:
        if MODEL_LOADED and model is not None:
            char = predict_character(canvas, model)
            st.session_state.text_output += char
        st.session_state.points.clear()
        st.session_state.canvas[:] = 0
    
    # Update session canvas
    st.session_state.canvas = canvas
    
    # Merge canvas with frame
    mask = canvas.sum(axis=2) > 0
    frame[mask] = canvas[mask]
    
    # Add text overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, f"Text: {st.session_state.text_output}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add gesture indicator
    if gesture:
        cv2.putText(frame, f"Gesture: {gesture}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Add color indicator
    cv2.circle(frame, (w - 30, 30), 15, draw_color, -1)
    cv2.circle(frame, (w - 30, 30), 16, (255, 255, 255), 1)
    
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), gesture

# ===============================
# UI
# ===============================
st.title("✍️ AI-Based Air Writing")
st.markdown("Write in the air using hand gestures!")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Color picker
    color = st.color_picker("Choose drawing color", "#FF00FF")
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    st.session_state.draw_color = (b, g, r)
    
    st.markdown("---")
    st.subheader("Instructions")
    st.markdown("""
    - 👆 **Index finger only**: Write
    - ✌️ **Index + Middle finger**: Space
    - ✊ **Fist (all fingers down)**: Delete
    - 📷 Use camera input below
    """)
    
    st.markdown("---")
    if st.button("Clear All"):
        st.session_state.canvas[:] = 0
        st.session_state.points.clear()
        st.session_state.text_output = ""
        st.rerun()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Camera Input")
    
    # Camera input
    img_file = st.camera_input("Take a picture with your hand gesture")
    
    if img_file is not None:
        # Read image
        image = Image.open(img_file)
        img_array = np.array(image)
        
        # Process frame
        processed_frame, gesture = process_frame(img_array, st.session_state.draw_color)
        
        # Display processed frame
        st.image(processed_frame, caption="Processed Frame", use_container_width=True)
        
        if gesture:
            st.info(f"Detected gesture: **{gesture}**")

with col2:
    st.subheader("Output")
    st.text_area("Recognized Text", st.session_state.text_output, height=200)
    
    if MODEL_LOADED:
        st.success("✅ Model loaded successfully")
    else:
        st.warning("⚠️ Model not loaded - Running in draw-only mode")
    
    st.markdown("---")
    st.subheader("Canvas")
    if st.session_state.canvas.sum() > 0:
        st.image(st.session_state.canvas, caption="Drawing Canvas", use_container_width=True)
    else:
        st.info("Canvas is empty")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit, MediaPipe, and TensorFlow")
