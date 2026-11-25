
import cv2
import numpy as np
from pathlib import Path


class FrameExtractor:

    def __init__(self, num_frames=16, image_size=224):
        self.num_frames = num_frames
        self.image_size = image_size

    def extract(self, video_path, output_dir):

        video_path = Path(video_path)
        output_dir = Path(output_dir)

        if not video_path.is_file():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(
                f"Unable to open video: {video_path}"
            )

        try:
            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            if total_frames <= 0:
                raise ValueError("Video contains no readable frames")

            # Select evenly spaced frame indices
            indices = np.linspace(
                0,
                total_frames - 1,
                self.num_frames,
                dtype=int
            )

            saved_frames = []

            for i, frame_index in enumerate(indices):

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    int(frame_index)
                )

                success, frame = cap.read()

                if not success:
                    continue

                # Resize frame
                frame = cv2.resize(
                    frame,
                    (self.image_size, self.image_size)
                )

                # Convert BGR to RGB
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # Convert RGB to BGR for OpenCV saving
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_RGB2BGR
                )

                frame_name = f"frame_{i:03d}.jpg"
                frame_path = output_dir / frame_name

                success = cv2.imwrite(
                    str(frame_path),
                    frame
                )

                if success:
                    saved_frames.append(str(frame_path))

            return saved_frames

        finally:
            cap.release()


if __name__ == "__main__":

    extractor = FrameExtractor(
        num_frames=16,
        image_size=224
    )

    print("FrameExtractor initialized successfully!")
    print("Frames per video:", extractor.num_frames)
    print("Image size:", extractor.image_size)
