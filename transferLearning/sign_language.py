import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model.h5')

class_names = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
       'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
       'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

#hand_detector = cv2.CascadeClassifier('haarcascade_hand.xml')

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


#    img = cv2.resize(frame, dsize=None, fx=1.0, fy=1.0)
    #gray = np.float32(img)
#    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cascade 얼굴 탐지 알고리즘
#    results = hand_detector.detectMultiScale(gray,  # 입력 이미지
#                                       scaleFactor=1.1,  # 이미지 피라미드 스케일 factor
#                                       minNeighbors=5,  # 인접 객체 최소 거리 픽셀
#                                       minSize=(60, 60)  # 탐지 객체 최소 크기
#                                       )
#    for box in results:
#        x, y, w, h = box
#        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
#        roi_gray = gray[y:y + h, x:x + w]
#        roi_color = img[y:y + h, x:x + w]

    cv2.imshow('RPS', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()