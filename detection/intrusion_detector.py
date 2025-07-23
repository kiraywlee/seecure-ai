# intrusion_detector.py
import cv2
import mediapipe as mp
import numpy as np
import os

mp_face_mesh = mp.solutions.face_mesh

FACE_PATH = "models/user_face.npy"
EYE_PATH = "models/user_eye_pos.npy"
GAZE_TOLERANCE = 40

def load_reference():
    if not os.path.exists(FACE_PATH) or not os.path.exists(EYE_PATH):
        print("[ERROR] 얼굴 등록 정보가 없습니다. face_register.py 먼저 실행하세요.")
        exit()

    face = np.load(FACE_PATH, allow_pickle=True)
    eye = np.load(EYE_PATH, allow_pickle=True)
    return face, eye

def get_eye_center(landmarks, shape):
    h, w = shape[:2]
    left = np.array([landmarks[468].x * w, landmarks[468].y * h])
    right = np.array([landmarks[473].x * w, landmarks[473].y * h])
    return ((left + right) / 2).astype(np.float32)

def is_same_person(face1, face2):
    # 픽셀 기반 유사도 계산 (향후 개선 가능)
    face1 = cv2.resize(face1, (100, 100))
    face2 = cv2.resize(face2, (100, 100))
    diff = np.mean(np.abs(face1.astype("float32") - face2.astype("float32")))
    return diff < 30  # 임계값 튜닝 가능

def main():
    user_face, user_eye = load_reference()
    cap = cv2.VideoCapture(0)
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    print("[INFO] 침입자 감지 시스템 실행 중 (ESC 종료)")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            faces = result.multi_face_landmarks
            if len(faces) > 1:
                print("[경고] 기준 사용자 외 얼굴 탐지됨 → 침입자 감지")

            for i, face_landmarks in enumerate(faces):
                eye_center = get_eye_center(face_landmarks.landmark, frame.shape)
                x, y, w_, h_ = 100, 100, 100, 100
                crop_face = cv2.resize(frame, (100, 100))

                if is_same_person(user_face, crop_face):
                    # 기준 사용자
                    dist = np.linalg.norm(eye_center - user_eye)
                    if dist < GAZE_TOLERANCE:
                        print("[정상] 사용자 화면 응시 중")
                    else:
                        print("[주의] 사용자 시선 이탈")
                else:
                    # 침입자
                    dist = np.linalg.norm(eye_center - user_eye)
                    if dist < GAZE_TOLERANCE:
                        print("[보안 경고] 침입자가 화면을 보고 있습니다!")
                    else:
                        print("[INFO] 침입자 감지 (시선은 다른 곳)")

        cv2.imshow("Intrusion Detector", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
