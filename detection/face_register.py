# face_register.py
import cv2
import mediapipe as mp
import numpy as np
import os

SAVE_FACE_PATH = "models/user_face.npy"
SAVE_EYE_PATH = "models/user_eye_pos.npy"

mp_face_mesh = mp.solutions.face_mesh

def main():
    cap = cv2.VideoCapture(0)
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    print("[INFO] 얼굴 등록 시작: 얼굴을 정면으로 바라보세요")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape

            # 기준 눈 중심 좌표 계산 (468: 오른쪽, 473: 왼쪽 홍채 중심)
            left = np.array([landmarks[468].x * w, landmarks[468].y * h])
            right = np.array([landmarks[473].x * w, landmarks[473].y * h])
            eye_center = ((left + right) / 2).astype(np.float32)

            # 얼굴 전체 이미지 저장
            face_img = cv2.resize(frame, (100, 100))  # 간단화용 리사이즈
            np.save(SAVE_FACE_PATH, face_img)
            np.save(SAVE_EYE_PATH, eye_center)

            print(f"[완료] 얼굴과 시선 좌표 저장됨: {eye_center}")
            break

        cv2.imshow("Face Register", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
