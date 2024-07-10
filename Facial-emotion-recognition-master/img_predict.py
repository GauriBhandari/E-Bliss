import cv2
import argparse
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

# Parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('image', help='path to input image file')
args = vars(ap.parse_args())

# Load model from JSON file
json_file = open('top_models\\fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load weights and them to model
model.load_weights('top_models\\fer.h5')

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread(args['image'])
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces_detected = classifier.detectMultiScale(gray_img, 1.18, 5)

for (x, y, w, h) in faces_detected:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi_gray = gray_img[y:y + w, x:x + h]
    roi_gray = cv2.resize(roi_gray, (48, 48))
    img_pixels = image.img_to_array(roi_gray)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255.0

    predictions = model.predict(img_pixels)
    max_index = int(np.argmax(predictions))

    emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
    predicted_emotion = emotions[max_index]

    cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

resized_img = cv2.resize(img, (1024, 768))
cv2.imshow('Facial Emotion Recognition', resized_img)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()



# import cv2
# import argparse
# import numpy as np
# from keras.models import model_from_json
# from keras.preprocessing import image
# import requests
# from io import BytesIO

# # Parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument('type', help='Type of input: "file" or "url"')
# ap.add_argument('path', help='Path to input image file or URL')
# args = vars(ap.parse_args())

# # Load model from JSON file
# json_file = open('top_models\\fer.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)

# # Load weights and them to model
# model.load_weights('top_models\\fer.h5')

# classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# if args['type'] == 'file':
#     # Load image from local file
#     img = cv2.imread(args['path'])
# elif args['type'] == 'url':
#     # Download image from URL
#     response = requests.get(args['path'])
#     img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
# else:
#     print('Invalid input type. Please provide either "file" or "url".')
#     exit()

# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces_detected = classifier.detectMultiScale(gray_img, 1.18, 5)

# for (x, y, w, h) in faces_detected:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     roi_gray = gray_img[y:y + w, x:x + h]
#     roi_gray = cv2.resize(roi_gray, (48, 48))
#     img_pixels = image.img_to_array(roi_gray)
#     img_pixels = np.expand_dims(img_pixels, axis=0)
#     img_pixels /= 255.0

#     predictions = model.predict(img_pixels)
#     max_index = int(np.argmax(predictions))

#     emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
#     predicted_emotion = emotions[max_index]

#     cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

# resized_img = cv2.resize(img, (1024, 768))
# cv2.imshow('Facial Emotion Recognition', resized_img)

# if cv2.waitKey(0) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()


# from flask import Flask, render_template, request, jsonify

# app = Flask(__name__)

# # Define a route to render the index.html page
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Define a route to handle the AJAX request from the frontend
# @app.route('/start_webcam', methods=['POST'])
# def start_webcam():
#     # Get the input type and path from the request
#     data = request.get_json()
#     input_type = data['type']
#     input_path = data['path']
    
#     # Add your emotion recognition code here based on the input type and path
#     # Replace this with your actual emotion recognition code
    
#     # Example response
#     response = {'message': 'Emotion recognition started successfully!'}
    
#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, request, render_template_string
# import cv2
# import numpy as np
# from keras.models import model_from_json
# from keras.preprocessing import image
# import requests
# from io import BytesIO
# import base64;

# app = Flask(__name__)

# # Load model from JSON file
# with open('top_models/fer.json', 'r') as json_file:
#     loaded_model_json = json_file.read()
# model = model_from_json(loaded_model_json)

# # Load weights into the model
# model.load_weights('top_models/fer.h5')

# classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# @app.route('/')
# def home():
#     return render_template_string(open('index.html').read())

# @app.route('/provide_url', methods=['POST'])
# def provide_url():
#     image_url = request.form['url']
#     response = requests.get(image_url)
#     img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces_detected = classifier.detectMultiScale(gray_img, 1.18, 5)

#     for (x, y, w, h) in faces_detected:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         roi_gray = gray_img[y:y + w, x:x + h]
#         roi_gray = cv2.resize(roi_gray, (48, 48))
#         img_pixels = image.img_to_array(roi_gray)
#         img_pixels = np.expand_dims(img_pixels, axis=0)
#         img_pixels /= 255.0

#         predictions = model.predict(img_pixels)
#         max_index = int(np.argmax(predictions))

#         emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
#         predicted_emotion = emotions[max_index]

#         cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

#     _, buffer = cv2.imencode('.jpg', img)
#     img_str = base64.b64encode(buffer).decode('utf-8')

#     return f'<img src="data:image/jpeg;base64,{img_str}" />'

# if __name__ == '__main__':
#     app.run(debug=True)
