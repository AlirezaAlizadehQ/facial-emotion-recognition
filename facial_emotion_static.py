import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.preprocessing import image
from PIL import Image

model = model_from_json(open("model.json", "r").read())
model.load_weights('model.h5')
face_haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

for filename in os.listdir('custom'): # iterate in the directory to adresss each image
    case_name = filename
    case_image = cv2.imread('custom/'+str(filename))
    gray_img = cv2.cvtColor(case_image, cv2.COLOR_BGR2GRAY) # convert to grayscale
    faces_detected = face_haar.detectMultiScale(gray_img, 1.3, 5) # detect the facial region

    for (a, b, c, d) in faces_detected:
        cv2.rectangle(case_image, (a, b), (a + c, b + d), (255, 0, 0), thickness=9)
        gray_array = gray_img[b:b + c, a:a + d]
        gray_array = cv2.resize(gray_array, (48, 48))
        image_to_vec = image.img_to_array(gray_array)
        image_to_vec = np.expand_dims(image_to_vec, axis=0)
        image_to_vec /= 255
        predictions = model.predict(image_to_vec)
        target_index = int(np.argmax(predictions[0])) # argmax to choose the emotion with highest score

        label_list = ('happy', 'fear', 'disgust', 'angry', 'sad', 'surprise', 'neutral') # 7 classes of emotions
        case_predicted_emotion = label_list[target_index]

        cv2.putText(case_image, case_predicted_emotion, (int(a), int(b)), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 64, 255), 3)

    resized_img = cv2.resize(case_image, (1200, 800)) # resize the image back to its normal size
    cv2.imwrite('out_custom' + '/' + str(case_name) + '.JPG', resized_img )
