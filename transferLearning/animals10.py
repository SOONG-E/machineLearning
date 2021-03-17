import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('animal_model.h5')

animals = ["dog", "horse", "elephant", "butterfly", "chicken", "cat", "cow", "sheep", "spider", "squirrel"]

colors = { 0 : (255, 0, 0), 1 : (255, 51, 153), 3 : (204, 51, 255),
           4: (0, 102, 255), 5: (0, 255, 255), 6: (0, 255, 0),
           7 : (204, 255, 102), 8 : (153, 153, 102), 9: (153, 153, 102),
           2 : (255, 153, 51)}

file_path = 'C:/Users/LG/Downloads/anitest.mp4'

video = cv2.VideoCapture(file_path)

if not video.isOpened():
    print("Could not open webcam")
    exit()

width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = video.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
file_name = 'animals.avi'

out = cv2.VideoWriter(file_name, fourcc, fps, (int(width), int(height)))

while video.isOpened():

    status, frame = video.read()

    if not status:
        break

    img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    img = np.array(img) / 255.0
    result = model.predict(img[np.newaxis, ...])
    predicted_class = np.argmax(result[0], axis=-1)

    predicted_class_name = animals[predicted_class]

    cv2.putText(frame, predicted_class_name.title(), (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, colors[predicted_class], 3)

    cv2.imshow('RPS', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
out.release()
cv2.destroyAllWindows()