
import cv2
from pathlib import Path


class VideoReader:

    def __init__(self, video_path):
        self.video_path = str(video_path)

        if not Path(self.video_path).is_file():
            raise FileNotFoundError(
                f"Video not found: {self.video_path}"
            )

        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise ValueError(
                f"Unable to open video: {self.video_path}"
            )

    def get_metadata(self):

        fps = self.cap.get(cv2.CAP_PROP_FPS)

        total_frames = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        duration = (
            total_frames / fps if fps > 0 else 0
        )

        return {
            "video_path": self.video_path,
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_seconds": round(duration, 2)
        }

    def read_frame(self, frame_number):

        self.cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        success, frame = self.cap.read()

        if not success:
            return None

        return frame

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
