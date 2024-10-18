import torch
import pandas as pd
import cv2
from PIL import Image


class Ucf05Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, mode, transform=None, max_frames=32, frame_size=(224, 224)):
        self.data = pd.read_csv(csv_file)
        self.mode = mode
        self.transform = transform
        self.max_frames = max_frames
        self.frame_size = frame_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        frames = self._load_frames(video_path)
        frames = [self._process_frame(frame) for frame in frames]
        frames = self._pad_or_truncate_frames(frames)
        frames = torch.stack(frames, dim=0) 
        frames = frames.permute(1, 0, 2, 3)  
        label = torch.tensor(label, dtype=torch.long)

        return frames, label


    def _load_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        return frames
    
    def _process_frame(self, frame):
        frame = cv2.resize(frame, self.frame_size)
        frame = Image.fromarray(frame)
        if self.transform:
            frame = self.transform(frame)
        return frame

    def _pad_or_truncate_frames(self, frames):
        if len(frames) < self.max_frames:
            frames.extend([frames[-1]] * (self.max_frames - len(frames)))
        elif len(frames) > self.max_frames:
            frames = frames[:self.max_frames]
        return frames


