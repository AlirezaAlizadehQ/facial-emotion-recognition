import cv2
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.preprocessing import image
from PIL import Image
# from VideoCapture import Device
model = model_from_json(open('model.json', 'r').read())
model.load_weights('model.h5')
face_haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_haar = cv2.CascadeClassifier('haarcascade_eye.xml')
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

'''
cam = Device()
cam.saveSnapshot('image.jpg')
'''

video_capture = cv2.VideoCapture(0)

flag = True
while flag:
    var, image = video_capture.read()
    if not var:
        continue

    frame_to_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    facial_region = face_haar.detectMultiScale(frame_to_grayscale, 1.35, 5)
    for (a, b, c, d) in facial_region:
        cv2.rectangle(image, (a, b), (a + c, b + d), (255, 0, 0), thickness=9)
        gray_array= frame_to_grayscale[b:b + c, a:a + d]
        gray_array=cv2.resize(gray_array, (48, 48))
        image_to_vec = image.img_to_array(gray_array)
        image_to_vec = np.expand_dims(image_to_vec, axis=0)
        image_to_vec = image_to_vec / 255
        predictions = model.predict(image_to_vec)
        target_index = int(np.argmax(predictions[0]))
        predicted_emotion = emotion_labels[target_index]

        cv2.putText(image, predicted_emotion, (int(a), int(b)), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 64, 255), 4)

    resized_img = cv2.resize(image, (800, 400))
    '''
    plt.imshow(resized_img, cmap='RdPu')
    plt.show()
    flag = False
    '''
    cv2.imshow('predicted emotion', resized_img)

    if cv2.waitKey(10) == ord('e'):
        flag = False
video_capture.release()
cv2.destroyAllWindows()








