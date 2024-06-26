from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import mediapipe as mp
import uuid
import os
from collections import Counter
from pose_classification import build_model
from viewer import App
from sign_parser import process_image, process_results

if __name__=="__main__":
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_detection
    classifier = build_model()

    app = QApplication(sys.argv)
    a = App(mp_hands, mp_face, mp_drawing, classifier, process_results)
    a.show()
    sys.exit(app.exec_())