# # -*- coding: utf-8 -*-
#
#
import numpy as np
import cv2, queue, threading, time
from tensorflow.keras.models import model_from_json
import pygame

# load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

json_file2 = open('model2.json', 'r')
loaded_model2_json = json_file2.read()
json_file2.close()
loaded_model2 = model_from_json(loaded_model2_json)
loaded_model2.load_weights("model2.h5")
loaded_model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# bufferless VideoCapture
class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


eye_zero_count = 0  # record times of closing eyes
t_count = 0  # record the number of loop
cap = VideoCapture(0)
lose_face_time = 0
lose_eye_time = 0
while True:
    time.sleep(.5)  # simulate time between events
    frame = cap.read()

    # Load face cascade and hair cascade from local folder
    face_cascPath = 'haarcascade_frontalface_alt.xml'
    openeye_cascPath = 'haarcascade_eye_tree_eyeglasses.xml'
    left_eyePath = 'haarcascade_lefteye_2splits.xml'
    right_eyePath = 'haarcascade_righteye_2splits.xml'
    faceCascade = cv2.CascadeClassifier(face_cascPath)
    lefteyeCascade = cv2.CascadeClassifier(left_eyePath)
    righteyeCascade = cv2.CascadeClassifier(right_eyePath)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.1, 3)
    roi_gray = np.empty_like(gray)
    pygame.init()
    pygame.mixer.init()
    if len(faces) == 0:
        cv2.putText(frame, text='Warning!!!', org=(250, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2,
                    thickness=2, color=(0, 0, 255))
        counting_frame = int(cv2.CAP_PROP_FRAME_COUNT)
        lose_face_time += counting_frame
        if lose_face_time > 14:  # if face did not show for 2s, it plays warning sound
            pygame.mixer.music.load('beepsound.mp3')
            pygame.mixer.music.play()
    else:
        lose_face_time = 0
    # Draw a rectangle over the face, and detect eyes in faces
    for (x, y, w, h) in faces:
        t_count += 1
        if len(faces) == 1:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # ROI is region of interest with area having face inside it.
        roi_gray = gray[y:y + h, x:x + w]
        left_area = frame[y:y + h, x + int(w / 2):x + w]
        left_area_gray = gray[y:y + h, x + int(w / 2):x + w]
        right_area = frame[y:y + h, x:x + int(w / 2)]
        right_area_gray = gray[y:y + h, x:x + int(w / 2)]

        left_eye = lefteyeCascade.detectMultiScale(left_area_gray, 1.1, 3)
        right_eye = righteyeCascade.detectMultiScale(right_area_gray, 1.1, 3)
        if len(left_eye) == 0 and len(right_eye) == 0:
            eye_zero_count += 1

        for (lx, ly, lw, lh) in left_eye:
            if len(left_eye) == 1:
                left_crop = left_area_gray[ly:ly + lh, lx:lx + lw]
                left_crop = cv2.resize(left_crop, (70, 70))
                left_pixels = left_crop.reshape(1, 70, 70, 1)
                predictions_left = loaded_model2.predict(left_pixels)
                if predictions_left < 1:  # red box if eye closed
                    cv2.rectangle(left_area, (lx, ly), (lx + lw, ly + lh), (0, 0, 255), 2)
                    counting_frame = int(cv2.CAP_PROP_FRAME_COUNT)
                    lose_eye_time += counting_frame
                    if lose_eye_time > 14:
                        pygame.mixer.music.load('beepsound.mp3')
                        pygame.mixer.music.play()
                else:  # green box if eye open
                    cv2.rectangle(left_area, (lx, ly), (lx + lw, ly + lh), (0, 255, 0), 2)
                    lose_eye_time = 0

        for (ex, ey, ew, eh) in right_eye:
            if len(right_eye) == 1:
                right_crop = right_area_gray[ey:ey + eh, ex:ex + ew]
                right_crop = cv2.resize(right_crop, (70, 70))
                right_pixels = right_crop.reshape(1, 70, 70, 1)
                predictions_right = loaded_model2.predict(right_pixels)
                if predictions_right < 1:  # eye close
                    cv2.rectangle(right_area, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
                    counting_frame = int(cv2.CAP_PROP_FRAME_COUNT)
                    lose_eye_time += counting_frame
                    if lose_eye_time > 14:
                        pygame.mixer.music.load('beepsound.mp3')
                        pygame.mixer.music.play()
                else:
                    cv2.rectangle(right_area, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    lose_eye_time = 0
    if t_count == 2:
        if eye_zero_count > 1:
            pygame.mixer.music.load('beepsound.mp3')
            pygame.mixer.music.play()
            eye_zero_count = 0
        t_count = 0

    cv2.imshow("frame", frame)
    if chr(cv2.waitKey(10) & 0xFF) == 'q':
        break
        cv2.destroyAllWindows()
