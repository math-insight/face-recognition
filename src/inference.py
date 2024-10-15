import cv2
import numpy as np
import keras
from keras_preprocessing.image import img_to_array
from concurrent.futures import ThreadPoolExecutor, as_completed
import tensorflow as tf


def load_models():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        except RuntimeError as e:
            print(e)
    
    face_classifier = cv2.CascadeClassifier(r'./models/haarcascade_frontalface_default.xml')
    gender_model = keras.saving.load_model(r'./models/gender_model_3epochs.h5', custom_objects={'mse': keras.losses.mean_squared_error})
    emotion_model = keras.saving.load_model(r'./models/emotion_detection_model_50epochs.h5', custom_objects={'mse': keras.losses.mean_squared_error})
    age_model = keras.saving.load_model(r'./models/age_model_3epochs.h5', custom_objects={'mse': keras.losses.mean_squared_error})
    
    return {
        'face_classifier': face_classifier,
        'emotion_model': emotion_model,
        'age_model': age_model,
        'gender_model': gender_model
    }


def process_face(face_data, frame, emotion_model, gender_model, age_model):
    x, y, w, h = face_data
    class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    gender_labels = ['Male', 'Female']

    roi_gray = cv2.resize(cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY), (48, 48))
    roi_color = cv2.resize(frame[y:y+h, x:x+w], (200, 200))

    # Emotion prediction
    roi_emotion = img_to_array(roi_gray) / 255.0
    roi_emotion = np.expand_dims(roi_emotion, axis=0)
    emotion_label = class_labels[np.argmax(emotion_model.predict(roi_emotion)[0])]

    # Gender prediction
    roi_gender = np.expand_dims(roi_color, axis=0)
    gender_label = gender_labels[int(gender_model.predict(roi_gender)[0][0] >= 0.5)]

    # Age prediction
    age = round(age_model.predict(roi_gender)[0][0])

    return (x, y, w, h, emotion_label, gender_label, age)


def classify(frame, face_classifier, emotion_model, age_model, gender_model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_face = {executor.submit(process_face, face, frame, emotion_model, gender_model, age_model): face for face in faces}
        
        for future in as_completed(future_to_face):
            x, y, w, h, emotion_label, gender_label, age = future.result()
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, gender_label, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Age={age}", (x+h, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

