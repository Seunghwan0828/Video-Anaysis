import torch
import torch.nn as nn
from torchvision import transforms
import cv2
from PIL import Image
import os
import numpy as np  
from src.model import CustomMViTv2Model

device = "cuda" if torch.cuda.is_available() else "cpu"


model = CustomMViTv2Model(num_classes=8, model_variant='mvit_base_32x3', pretrained=False)
model = model.to(device)


checkpoint_path = 'src/checkpoints/aihub/best_modelv2_aihub_final_epoch_18.pth'

checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint


new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("module."):
        new_key = key[len("module."):]
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

model.load_state_dict(new_state_dict)
model.eval()

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_video_frames(video_path, transform, max_frames=32, frame_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, frame_size)
        frame = Image.fromarray(frame)
        if transform:
            frame = transform(frame)
        frames.append(frame)

    cap.release()

    if len(frames) < max_frames:
        frames.extend([frames[-1]] * (max_frames - len(frames)))
    elif len(frames) > max_frames:
        frames = frames[:max_frames]

    frames = torch.stack(frames, dim=0)
    frames = frames.permute(1, 0, 2, 3) 
    return frames.unsqueeze(0) 

def inference(video_path):
    frames = load_video_frames(video_path, test_transforms).to(device)
    with torch.no_grad():
        outputs = model(frames)
        # print(outputs)
        probabilities = nn.Softmax(dim=1)(outputs)
        _, preds = torch.max(outputs, 1)
    return preds.item(), probabilities

video_path = 'data/aihub_test_videos/situp.mp4'
prediction, probabilities = inference(video_path)
print(f'Predicted class: {prediction}')

class_names = {
    0: "walking",
    1: "running",
    2: "sitting",
    3: "collapse",
    4: "jump",
    5: "climbing stairs",
    6: "push up",
    7: "sit up"
}



if prediction in class_names:
    print(f'Predicted class name: {class_names[prediction]}')


print("Class probabilities:")
for i, prob in enumerate(probabilities.squeeze()):
    percentage = prob.item() * 100  
    print(f"Class '{class_names[i]}': Probability = {percentage:.2f}%")

