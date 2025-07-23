# Seecure AI : 실시간 시선 기반 침입자 감지 시스템 ver.01
> Edge AI 기반 보안 시스템



## 주요 기능
- 얼굴 등록
  MediaPipe FaceMEsh 기반 사용자 얼굴 및 시선 좌표 등록
- 시선 추적
  사용자가 화면을 응시 중인지 실시간으로 추적
- 침입자 감지
  프레임 내 다수의 인물이 있을 경우, 등록된 사용자 외 인물을 탐지  
  침입자의 시선이 화면을 향하면 보안 경고



## 디렉토리 구조
seecure_final_git/
├── detection/
│ ├── face_register.py # 얼굴 및 기준 시선 등록
│ ├── gaze_tracker.py # 시선 추적 (실시간)
│ ├── intrusion_detector.py # 제3자 감지 및 경고
│ └── utils.py # 공통 유틸 함수
├── models/
│ ├── user_face.npy # 등록된 얼굴 이미지 (100x100)
│ └── user_eye_pos.npy # 기준 시선 좌표 (x, y)
├── test_face_register.py # 얼굴 등록 테스트용
├── test_gaze_tracker.py # 시선 추적 테스트용
├── test_intrusion.py # 침입자 감지 테스트용
└── .gitignore



## 기술 스택
| 기술        | 설명                                                                 |
|-------------|----------------------------------------------------------------------|
| **Python 3.10+** | 전체 시스템 구현 언어                                              |
| **MediaPipe**    | 얼굴/시선 탐지 (Face Mesh 기반 iris landmark 활용)               |
| **OpenCV**       | 실시간 영상 처리 및 이미지 추출                                   |
| **NumPy**        | 얼굴 데이터 및 시선 좌표 저장, 거리 계산 등                       |
| **Face Comparison (임시)** | 픽셀 기반 평균 절대 오차 계산으로 사용자 얼굴 유사도 측정 |
| **기타**         | 추후 YOLOv8-face, InsightFace, face_recognition 등 연동 고려 중 |

---



## 보완이 필요한 점
1. 침입자 얼굴 구분 정확도
   - 현재는 단순 픽셀 기반 유사도 (np.mean(abs(diff)))를 사용하고 있어 조명, 각도, 화질 변화에 매우 민감
3. 시선 기준 거리 고정값 (40px)
   - 해상도에 상관없이 동일한 임계값을 적용하기 때문에 저/고해상도에서 일관성 부족
4. 멀티페이스 상황에서의 오탐/누락
   - 2명 이상 얼굴이 감지될 경우 침입자 판단 로직은 존재하지만, 정확도 낮음



## 고려중인 개선 방향
- FaceNet 기반 얼굴 임베딩 도입
- YOLOv8-face + Gaze Regression 결합 시스템 구축
- 시선 + 사용자 인증 결합 기반 이중 보안
- 모바일/Edge 디바이스 대응을 위한 TFLite 최적화
