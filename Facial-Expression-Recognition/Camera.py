import cv2
from tensorflow.keras.models import model_from_json
import numpy as np

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = model_from_json(open('model.json', 'r').read())
model.load_weights('model_weights.h5')
font = cv2.FONT_HERSHEY_COMPLEX

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, fr = cap.read()
    if cv2.waitKey(1) == ord('q') or ret == False: 
        break 
    
    gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_fr, 1.1, 4)

    for (x, y, w, h) in faces:
        fc = gray_fr[y:y+h, x:x+w]

        roi = cv2.resize(fc, (48, 48))
        pred = model.predict(roi[np.newaxis, :, :, np.newaxis])
        max_index = np.argmax(pred[0])
        emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad','Surprise')
        emotions = emotions[max_index]
        emotions = emotions + ': ' + str(round(pred[0][max_index]*100,2))

        cv2.putText(fr, emotions, (x, y-7), font, 0.8, (255, 0, 0), 2)
        cv2.rectangle(fr,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow('img', fr)

cap.release()
cv2.destroyAllWindows()
