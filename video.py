from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QScrollArea, QLineEdit
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import mediapipe as mp
import uuid
import os
import datetime
from util import get_frequent_output, get_text

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, str, str, str)

    def __init__(self, mp_hands, mp_drawing, classifier, process_image):
        super().__init__()
        self._run_flag = True
        self.mp_hands = mp_hands
        self.mp_drawing = mp_drawing
        self.classifier = classifier
        self.process_image = process_image

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        output_list = np.array([])

        korean_text = ''
        english_text = ''
        past_sentence = ''
        last_frame = 0
        frame_count = 0

        with self.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
            while self._run_flag:
                ret, frame = cap.read()
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.flip(image, 1)
                image.flags.writeable = False
                
                # 이미지 frame을 mediapipe로 넘겨 손 판별
                results = hands.process(image)
                
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                output_list = []
            
                if results.multi_hand_landmarks: 
                    # 손이 인식된 경우 좌표가 multi_hand_landmarks에 저장됨. 손이 화면에 없을 경우 if문 skip
                    for num, hand in enumerate(results.multi_hand_landmarks):
                        # HAND_CONNECTIONS: 관절 포인트 몇번몇번이 연결되어 있는지 정보
                        self.mp_drawing.draw_landmarks(image, hand, self.mp_hands.HAND_CONNECTIONS,
                                                 self.mp_drawing.DrawingSpec(color=(255, 2, 102), thickness=2, circle_radius=4),
                                                 self.mp_drawing.DrawingSpec(color=(255, 222, 4), thickness=2, circle_radius=2)) 

                    if len(results.multi_hand_landmarks) == 2:
                        output, prob = self.process_image(results, self.classifier)
                        if prob >= 7.0:
                            output_list.append(output)
                        else:
                            output_list.append(-1)
                        past_sentence = ''
                        if frame_count > last_frame:
                            if korean_text != '':
                                past_sentence = korean_text + " ( " +english_text + " )"
                            korean_text = ''
                            english_text = ''
                            last_frame = 0
                            if frame_count % 30 == 0:
                                minimal = int(len(output_list) * 0.2)
                                freq_output, freq_count = get_frequent_output(output_list)
                                if freq_count >= minimal and freq_output != -1:
                                    korean_text, english_text = get_text(freq_output)
                                    last_frame = frame_count + 60
                self.change_pixmap_signal.emit(image, korean_text, english_text, past_sentence)
                frame_count += 1
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()