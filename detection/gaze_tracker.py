# detection/gaze_tracker.py

import cv2
import mediapipe as mp
import numpy as np
import os
import time

GAZE_REF_PATH = "models/user_eye_pos.npy"

mp_face_mesh = mp.solutions.face_mesh

class GazeTracker:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                               max_num_faces=1,
                                               refine_landmarks=True)
        self.last_log_time = 0 # 로그 시간 기록용
        self.ref_point = None

        if os.path.exists(GAZE_REF_PATH):
            self.ref_point = np.load(GAZE_REF_PATH)
            print(f"[로드 완료] 기준 시선 좌표 로드됨: {self.ref_point}")
        else:
            print("[경고] 기준 시선 좌표가 없습니다. 먼저 얼굴 등록을 진행하세요.")

    def get_eye_center(self, landmarks, image_shape):
        # 양쪽 눈 중앙의 평균을 기준 시선으로 사용 (478, 473 - 왼눈, 468, 468 - 오른눈 중앙)
        ih, iw = image_shape[:2]
        left_eye = landmarks[473]
        right_eye = landmarks[468]

        left = np.array([left_eye.x * iw, left_eye.y * ih])
        right = np.array([right_eye.x * iw, right_eye.y * ih])

        center = (left + right) / 2
        return center

    def is_gaze_within_tolerance(self, current_point, tolerance=40):
        if self.ref_point is None:
            return False
        dist = np.linalg.norm(current_point - self.ref_point)
        return dist < tolerance

    def track(self):
        cap = cv2.VideoCapture(0)
        print("시선 추적 시작 (ESC로 종료)")
        self.last_log_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.face_mesh.process(rgb)

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark
                eye_center = self.get_eye_center(landmarks, frame.shape)

                if self.ref_point is not None:
                    current_time = time.time()
                    if self.ref_point is not None and (current_time - self.last_log_time > 0.5): # 0.5초마다 로그 표시
                        if self.is_gaze_within_tolerance(eye_center):
                            x, y = int(eye_center[0]), int(eye_center[1])
                            print("[응시 중] 사용자가 화면을 응시 중입니다.") # 좌표 추가 예정
                        else:
                            print("[경고] 시선 이탈 감지")
                        self.last_log_time = current_time

                # 디버깅용 시각화
                cv2.circle(frame, tuple(eye_center.astype(int)), 5, (0,255,0), -1)
                if self.ref_point is not None:
                    cv2.circle(frame, tuple(self.ref_point.astype(int)), 5, (255,0,0), -1)

            cv2.imshow("Gaze Tracker", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = GazeTracker()
    tracker.track()
