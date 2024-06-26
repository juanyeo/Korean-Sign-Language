from PyQt5 import QtGui, QtCore
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
from video import VideoThread
from sign_parser import process_results

class App(QWidget):
    def __init__(self, mp_hands, mp_face, mp_drawing, classifier, process_image):
        super().__init__()
        self.setWindowTitle("Korean Sign Language Translator")
        self.setStyleSheet("background-color: rgb(246, 246, 244);")
        self.disply_width = 1000
        self.display_height = 800

        # 메인 레이아웃
        hbox = QHBoxLayout(self)

        # 비디오와 레이블을 위한 레이아웃
        vbox_left = QVBoxLayout()
        self.image_label = QLabel(self) 
        self.image_label.resize(self.disply_width, self.display_height)
        self.image_label.setStyleSheet("padding-bottom: 0px; margin-bottom: 0px")

        self.korean_subtitle = QLabel("구급차 불러주세요")
        self.korean_subtitle.resize(self.disply_width, 60)
        self.korean_subtitle.setFixedHeight(60)
        self.korean_subtitle.setAlignment(QtCore.Qt.AlignCenter)
        self.korean_subtitle.setStyleSheet("background-color: white; font-size: 40pt;")
        # self.subtitle.setGeometry(QtCore.QRect(1000, 400, 1100, 200))

        self.english_subtitle = QLabel("Call ambulance")
        self.english_subtitle.resize(self.disply_width, 30)
        self.english_subtitle.setFixedHeight(30)
        self.english_subtitle.setAlignment(QtCore.Qt.AlignCenter)
        self.english_subtitle.setStyleSheet("background-color: white; font-size: 30pt;")

        self.textLabel = QLabel('Korean Sign Language Translator\n한국어 수화 통역 서비스')
        vbox_left.addWidget(self.image_label)
        vbox_left.addWidget(self.korean_subtitle)
        vbox_left.addWidget(self.english_subtitle)
        vbox_left.addWidget(self.textLabel)
        hbox.addLayout(vbox_left)

        # 오른쪽 UI를 위한 레이아웃
        vbox_right = QVBoxLayout()

        # 대학교 마크 추가
        self.univ_logo_label = QLabel(self)
        pixmap = QPixmap('assets/image.png')  # 이미지 파일 경로 설정
        original_width = pixmap.width()
        original_height = pixmap.height()
        scaled_width = original_width // 2
        scaled_height = original_height // 2
        scaled_pixmap = pixmap.scaled(scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)  # 원본 크기의 50%로 조절 및 부드러운 변환
        self.univ_logo_label.setPixmap(scaled_pixmap)
        self.univ_logo_label.setAlignment(Qt.AlignCenter)
        vbox_right.addWidget(self.univ_logo_label)

        # 채팅창을 위한 스크롤 영역
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.chat_widget = QWidget()
        self.chat_widget.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.chat_layout = QVBoxLayout(self.chat_widget)
        self.chat_layout.addStretch()
        self.scroll_area.setWidget(self.chat_widget)
        vbox_right.addWidget(self.scroll_area)

        # 사용자 입력 필드와 버튼 추가
        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("메시지를 입력하세요")
        self.add_button = QPushButton("메시지 추가")
        self.add_button.clicked.connect(self.add_chat_message)
        vbox_right.addWidget(self.input_field)
        vbox_right.addWidget(self.add_button)

        hbox.addLayout(vbox_right)

        self.setLayout(hbox)

        # create the video capture thread
        self.thread = VideoThread(mp_hands, mp_drawing, classifier, process_image)
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray, str, str, str)
    def update_image(self, cv_img, korean_text, english_text, past_sentence):
        """Updates the image_label with a new opencv image"""
        # output = process_results(results, classifier)
        # print(output)
        if past_sentence != '':
            self.add_past_message(past_sentence)
        self.korean_subtitle.setText(korean_text)
        self.english_subtitle.setText(english_text)
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
    def add_past_message(self, past_message):
        """Adds a chat message to the chat layout"""
        message = past_message
        if message:
            current_time = datetime.datetime.now().strftime("%H:%M:%S")

            # 메시지와 시간을 담을 위젯
            message_widget = QWidget()
            message_layout = QVBoxLayout(message_widget)
            message_layout.setContentsMargins(0, 0, 0, 0)
            message_layout.setSpacing(2)

            # 메시지 레이블
            message_label = QLabel(message)
            message_label.setStyleSheet("""
                background-color: #E0F7FA;  /* 연한 파란색 배경 */
                border: 1px solid #B2EBF2;  /* 연한 파란색 테두리 */
                border-radius: 10px;        /* 둥근 테두리 */
                padding: 5px;               /* 안쪽 여백 */
            """)
            message_label.setWordWrap(True)  # 텍스트 줄바꿈 허용

            # 시간 레이블
            time_label = QLabel(current_time)
            time_label.setStyleSheet("color: gray; font-size: 10px;")
            time_label.setAlignment(Qt.AlignRight)

            # 메시지와 시간을 레이아웃에 추가
            message_layout.addWidget(message_label)
            message_layout.addWidget(time_label)

            #안녕하세요 밑에 이모지 추가
            if message == '안녕하세요 ( Hello )': 
                emoji_label = QLabel(self)
                emoji_pixmap = QPixmap('assets/hello.png')
                original_width = emoji_pixmap.width()
                original_height = emoji_pixmap.height()
                scaled_width = original_width // 16
                scaled_height = original_height // 16
                scaled_pixmap = emoji_pixmap.scaled(scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                emoji_label.setPixmap(scaled_pixmap)
                emoji_label.setAlignment(Qt.AlignLeft)
                message_layout.addWidget(emoji_label)
            
            # 감사합니다 밑에 이모지 추가
            if message == '감사합니다 ( Thank you )' : 
                emoji_label = QLabel(self)
                emoji_pixmap = QPixmap('assets/thank.png')
                original_width = emoji_pixmap.width()
                original_height = emoji_pixmap.height()
                scaled_width = original_width // 16
                scaled_height = original_height // 16
                scaled_pixmap = emoji_pixmap.scaled(scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                emoji_label.setPixmap(scaled_pixmap)
                emoji_label.setAlignment(Qt.AlignLeft)
                message_layout.addWidget(emoji_label)

            self.chat_layout.addWidget(message_widget)
            # Scroll to the bottom
            self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())
            self.input_field.clear()

    def add_chat_message(self):
        """Adds a chat message to the chat layout"""
        message = self.input_field.text()
        if message:
            current_time = datetime.datetime.now().strftime("%H:%M:%S")

            # 메시지와 시간을 담을 위젯
            message_widget = QWidget()
            message_layout = QVBoxLayout(message_widget)
            message_layout.setContentsMargins(0, 0, 0, 0)
            message_layout.setSpacing(2)

            # 메시지 레이블
            message_label = QLabel(message)
            message_label.setStyleSheet("""
                background-color: #E0F7FA;  /* 연한 파란색 배경 */
                border: 1px solid #B2EBF2;  /* 연한 파란색 테두리 */
                border-radius: 10px;        /* 둥근 테두리 */
                padding: 5px;               /* 안쪽 여백 */
            """)
            message_label.setWordWrap(True)  # 텍스트 줄바꿈 허용

            # 시간 레이블
            time_label = QLabel(current_time)
            time_label.setStyleSheet("color: gray; font-size: 10px;")
            time_label.setAlignment(Qt.AlignRight)

            # 메시지와 시간을 레이아웃에 추가
            message_layout.addWidget(message_label)
            message_layout.addWidget(time_label)

            #안녕하세요 밑에 이모지 추가
            if message == '안녕하세요 ( Hello )' : 
                emoji_label = QLabel(self)
                emoji_pixmap = QPixmap('assets/hello.png')
                original_width = emoji_pixmap.width()
                original_height = emoji_pixmap.height()
                scaled_width = original_width // 16
                scaled_height = original_height // 16
                scaled_pixmap = emoji_pixmap.scaled(scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                emoji_label.setPixmap(scaled_pixmap)
                emoji_label.setAlignment(Qt.AlignLeft)
                message_layout.addWidget(emoji_label)
            
            # 감사합니다 밑에 이모지 추가
            if message == '감사합니다 ( Thank you )' : 
                emoji_label = QLabel(self)
                emoji_pixmap = QPixmap('assets/thank.png')
                original_width = emoji_pixmap.width()
                original_height = emoji_pixmap.height()
                scaled_width = original_width // 16
                scaled_height = original_height // 16
                scaled_pixmap = emoji_pixmap.scaled(scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                emoji_label.setPixmap(scaled_pixmap)
                emoji_label.setAlignment(Qt.AlignLeft)
                message_layout.addWidget(emoji_label)

            self.chat_layout.addWidget(message_widget)
            # Scroll to the bottom
            self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())
            self.input_field.clear()