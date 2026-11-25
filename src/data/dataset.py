
import cv2
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import Dataset


class VideoFrameDataset(Dataset):

    def __init__(
        self,
        samples,
        processor,
        num_frames=16,
        training=False
    ):
        self.samples = samples
        self.processor = processor
        self.num_frames = num_frames
        self.training = training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        frame_dir, label = self.samples[index]
        frame_dir = Path(frame_dir)

        frame_paths = sorted(
            frame_dir.glob("frame_*.jpg")
        )

        if not frame_paths:
            raise ValueError(
                f"No frames found in {frame_dir}"
            )

        indices = np.linspace(
            0,
            len(frame_paths) - 1,
            self.num_frames,
            dtype=int
        )

        video_frames = []

        for frame_index in indices:

            frame_path = frame_paths[frame_index]
            frame = cv2.imread(str(frame_path))

            if frame is None:
                raise ValueError(
                    f"Unable to read {frame_path}"
                )

            processed = self.processor.preprocess(
                frame,
                training=self.training
            )

            processed = np.transpose(
                processed,
                (2, 0, 1)
            ).copy()

            tensor = torch.from_numpy(
                processed
            ).float()

            video_frames.append(tensor)

        video_tensor = torch.stack(video_frames)

        label_tensor = torch.tensor(
            label,
            dtype=torch.long
        )

        return video_tensor, label_tensor


if __name__ == "__main__":

    print("NeuroMotion AI Dataset module loaded!")
    print("PyTorch version:", torch.__version__)
    print("Dataset class:", VideoFrameDataset.__name__)
