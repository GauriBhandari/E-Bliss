from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

app = Flask(__name__)

# Load model from JSON file
json_file = open('top_models\\fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load weights and them to model
model.load_weights('top_models\\fer.h5')

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def predict_emotion(frame):
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.1, 6, minSize=(150, 150))

    for (x, y, w, h) in faces_detected:
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255.0

        predictions = model.predict(img_pixels)
        max_index = int(np.argmax(predictions))

        emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
        predicted_emotion = emotions[max_index]

        cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_webcam')
def start_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = predict_emotion(frame)

        cv2.imshow('Facial Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
   

if __name__ == '__main__':
    app.run(debug=True)


