from detection.gaze_tracker import GazeTracker

def test_gaze_tracking():
    print("[START] 시선 추적 테스트 시작")
    tracker = GazeTracker()
    tracker.track()

if __name__ == "__main__":
    test_gaze_tracking()