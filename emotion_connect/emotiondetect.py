# Inside emotion_detction.py
print("Script is running.")

from tensorflow import keras
from tensorflow.keras.models import Sequential
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import cv2
import numpy as np
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
				3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
emotion_model = keras.models.load_model(r"F:\OneDrive\Desktop\flak connect\emotion_model1.h5")
import cv2
import numpy as np
cap = cv2.VideoCapture(0)
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame is empty
    if not ret or frame is None:
        print("Error: Failed to capture frame")
        break

    # Resize the frame
    frame = cv2.resize(frame, (1280, 720))

    # Create a face detector
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces available on the camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Process each face available on the camera
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict the emotions (assuming emotion_model and emotion_dict are defined somewhere)
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)

    # Check for user input to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()