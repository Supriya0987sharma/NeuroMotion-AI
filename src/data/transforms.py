
import cv2
import numpy as np


class ImagePreprocessor:

    def __init__(self, image_size=224):

        self.image_size = image_size

        # ImageNet normalization statistics
        self.mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )

        self.std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

    def preprocess(self, frame, training=False):

        if frame is None:
            raise ValueError("Input frame is empty")

        # Resize image
        frame = cv2.resize(
            frame,
            (self.image_size, self.image_size)
        )

        # OpenCV BGR to RGB
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Data augmentation during training only
        if training:

            # Random horizontal flip
            if np.random.random() < 0.5:
                frame = cv2.flip(frame, 1)

            # Random brightness adjustment
            if np.random.random() < 0.3:

                factor = np.random.uniform(0.8, 1.2)

                frame = np.clip(
                    frame.astype(np.float32) * factor,
                    0,
                    255
                ).astype(np.uint8)

        # Convert to float
        frame = frame.astype(np.float32) / 255.0

        # Normalize RGB channels
        frame = (frame - self.mean) / self.std

        return frame


if __name__ == "__main__":

    # Create a dummy image for testing
    dummy_frame = np.random.randint(
        0, 256,
        (480, 640, 3),
        dtype=np.uint8
    )

    processor = ImagePreprocessor(image_size=224)

    processed = processor.preprocess(
        dummy_frame,
        training=True
    )

    print("Preprocessing successful!")
    print("Output shape:", processed.shape)
    print("Output dtype:", processed.dtype)
    print("Minimum value:", processed.min())
    print("Maximum value:", processed.max())
