# detection/utils.py

import mediapipe as mp
import numpy as np
import cv2

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# MediaPipe 얼굴 감지 객체 생성 (공용)
def create_face_detector(confidence=0.6):
    return mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=confidence)

# 얼굴 이미지 전처리 (100x100 resize 후 벡터화)
def preprocess_face(face_img):
    resized = cv2.resize(face_img, (100, 100))
    return resized.flatten()

# 유클리드 거리 계산 (얼굴 비교용)
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

# 얼굴 바운딩 박스 추출 함수 (MediaPipe Detection → 절대좌표 변환)
def get_absolute_bbox(detection, image_shape):
    bbox = detection.location_data.relative_bounding_box
    h, w, _ = image_shape
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    w_box = int(bbox.width * w)
    h_box = int(bbox.height * h)
    return x, y, w_box, h_box
