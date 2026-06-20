import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pygame

# ---------- Load Models ----------
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")
emotion_model = load_model("emotion_model.h5")
drowsiness_model = load_model("drowsiness_cnn.h5")

# ---------- Initialize Pygame for Alarm ----------
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")

# ---------- Variables ----------
eye_close_counter = 0
EYE_CLOSE_THRESHOLD = 3
emotion_labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

# ---------- Video Capture ----------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # ----------------- Emotion Detection -----------------
        face_resized = cv2.resize(roi_gray, (48,48))
        face_resized = face_resized.astype("float")/255.0
        face_resized = img_to_array(face_resized)
        face_resized = np.expand_dims(face_resized, axis=0)
        emotion_prediction = emotion_model.predict(face_resized, verbose=0)
        emotion_label = emotion_labels[np.argmax(emotion_prediction)]
        
        cv2.putText(frame, f"Emotion: {emotion_label}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        # ----------------- Eye / Drowsiness Detection -----------------
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
        eye_status = "Open"

        for (ex, ey, ew, eh) in eyes:
            eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_img = cv2.resize(eye_img, (32,32))
            eye_img = cv2.cvtColor(eye_img, cv2.COLOR_GRAY2RGB)
            eye_img = eye_img.astype("float")/255.0
            eye_img = img_to_array(eye_img)
            eye_img = np.expand_dims(eye_img, axis=0)

            prediction = drowsiness_model.predict(eye_img, verbose=0)
            if np.argmax(prediction) == 1:
                eye_status = "Closed"
                break
        
        # ----------------- Alarm Logic -----------------
        if eye_status == "Closed":
            eye_close_counter += 1
        else:
            eye_close_counter = 0

        if eye_close_counter >= EYE_CLOSE_THRESHOLD:
            cv2.putText(frame, "DROWSINESS ALERT!", (x, y+h+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play(-1)
        else:
            pygame.mixer.music.stop()
        
        cv2.putText(frame, f"Drowsiness: {eye_status}", (x, y+h+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    
    cv2.imshow("Driver Safety Monitoring", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
