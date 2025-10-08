"""Quick test of threaded capture"""

from src.capture.threaded_camera import ThreadedCamera, CameraConfig
from src.capture.fps_counter import FPSCounter
import cv2
import time


def test_threaded_capture():
    config = CameraConfig(
        src=2,  # Default webcam
        max_queue_size=5,
        width=640,
        height=480
    )

    fps_counter = FPSCounter(window_size=30)

    with ThreadedCamera(config) as camera:
        time.sleep(0.5)  # Let camera stabilize

        for _ in range(300):  # 10 seconds @ 30fps
            ret, frame = camera.read()

            if ret:
                fps_counter.update()
                cv2.imshow('Threaded Capture Test', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Print results
        stats = fps_counter.get_stats()
        cam_stats = camera.get_stats()

        print(f"\n=== Performance Metrics ===")
        print(f"Median FPS: {stats.fps_median:.1f}")
        print(f"P95 FPS: {stats.fps_p95:.1f}")
        print(f"Frame Time: {stats.frame_time_ms:.1f}ms")
        print(f"Frames Dropped: {cam_stats['frames_dropped']}")
        print(f"Drop Rate: {cam_stats['drop_rate']:.2%}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_threaded_capture()