import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model.h5')

class_names = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
       'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
       'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

while webcam.isOpened():

    status, frame = webcam.read()

    if not status:
        break

    img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    img = np.array(img) / 255.0
    result = model.predict(img[np.newaxis, ...])
    predicted_class = np.argmax(result[0], axis=-1)

    predicted_class_name = class_names[predicted_class]

    cv2.putText(frame, predicted_class_name.title(), (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 3)

    cv2.imshow('RPS', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()