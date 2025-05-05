import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model
from PIL import Image
import time
import matplotlib.pyplot as plt
from collections import deque

# Load the pre-trained model
model = load_model('output/best_model.h5')
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy",
                4: "Neutral", 5: "Sad", 6: "Surprised"}

# Global variables for emotion tracking
emotion_history = deque(maxlen=120)  # 120 data points (e.g., 2 minutes at 1 point per second)

def preprocess_image(frame):
    # Convert frame to RGB and resize to 96x96
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (96, 96))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def detect_faces(frame):
    # Use OpenCV's Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces

# Streamlit app
st.title("Classroom Engagement App")

# Layout: Two columns
col1, col2 = st.columns(2)

with col1:
    st.header("Presentation Panel")
    st.write("""
    ### Business Model: Commercialization Strategy
    1. **Customer Segments**: K-12 Institutions, Higher Education, Educators, Teachers, Administrators seeking real-time engagement analytics.
    - **Corporate Training Providers**: Companies like Coursera for Business, LinkedIn Learning.
    - **Healthcare**: Telehealth patient engagement.
    2. **Value Propositions**: Core value - Instant feedback on student engagement (boredom, confusion, focus/CDCPA).
    - **Unique Differentiation**: Edge AI Processing, Actionable Dashboards, Teachers receive dynamic lesson adjustments/recommendations.
    3. **Revenue Streams**: Subscription tiers (basic analytics, premium insights), licensing, cloud infrastructure costs.
    4. **Cost Structure**: AI development engineers, cloud training (AWS/GC), GDPR/COPPA audits, legal compliance.
    """)

# Initialize session state for webcam and engagement
if 'run' not in st.session_state:
    st.session_state.run = False
if 'cap' not in st.session_state:
    st.session_state.cap = cv2.VideoCapture(0)
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'avg_engagement_percent' not in st.session_state:
    st.session_state.avg_engagement_percent = 0

with col2:
    st.header("Classroom Engagement Dashboard")
    # Checkbox to start/stop emotion detection
    st.session_state.run = st.checkbox('Run Emotion Detection', value=st.session_state.run)
    FRAME_WINDOW = st.image([])
    GRAPH_WINDOW = st.empty()  # Placeholder for the graph

    # Process frames while the checkbox is checked
    if st.session_state.run:
        cap = st.session_state.cap
        ret, frame = cap.read()
        if not ret:
            st.write("Error: Could not access webcam.")
            st.session_state.run = False
            st.rerun()

        # Increment frame count for graph update frequency
        st.session_state.frame_count += 1

        # Detect faces and process emotions
        faces = detect_faces(frame)
        engagement_scores = []

        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            img_array = preprocess_image(roi)
            prediction = model.predict(img_array, verbose=0)
            emotion_index = np.argmax(prediction)
            emotion = emotion_dict[emotion_index]

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Calculate engagement score for this face
            engagement_score = 1 if emotion == "Happy" else 0 if emotion == "Neutral" else -1
            engagement_scores.append(engagement_score)

        # Calculate average engagement
        if engagement_scores:
            avg_engagement = np.mean(engagement_scores)
        else:
            avg_engagement = 0  # Default to neutral if no faces detected
            cv2.putText(frame, "No faces detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Update emotion history
        emotion_history.append(avg_engagement)

        # Calculate average engagement percentage
        st.session_state.avg_engagement_percent = max(0, min(100, (avg_engagement + 1) * 50))

        # Display the frame
        FRAME_WINDOW.image(frame, channels="BGR")

        # Update the graph every 10 frames (~1 second at 10 FPS)
        if st.session_state.frame_count % 10 == 0:
            fig, ax = plt.subplots()
            times = range(len(emotion_history))
            engagement_values = [(val + 1) * 50 for val in emotion_history]  # Scale to 0-100
            ax.plot(times, engagement_values, label="Emotion Trends")
            ax.set_title("Emotion Trends Over Time (Smoothed & Reversed Y-axis)")
            ax.set_xlabel("Time (minutes)")
            ax.set_ylabel("Engagement Level")
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_yticklabels(["Not Interested", "Bored", "Neutral", "Engaged", "Highly Engaged"])
            ax.legend()
            GRAPH_WINDOW.pyplot(fig)
            plt.close()

        # Add a small delay to control frame rate
        time.sleep(0.1)

        # Rerun the script to simulate live updates
        st.rerun()

    else:
        # Release the webcam when not running
        if st.session_state.cap.isOpened():
            st.session_state.cap.release()
        st.session_state.cap = cv2.VideoCapture(0)  # Reset for next run

# Display average engagement
with col2:
    st.write(f"Average Engagement: {st.session_state.avg_engagement_percent:.0f}%")