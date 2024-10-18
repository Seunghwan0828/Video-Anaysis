import torch
import torch.nn as nn
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
import threading
from src.model import CustomMViTv2Model
import time

device = "cuda" if torch.cuda.is_available() else "cpu"


model = CustomMViTv2Model(num_classes=400, model_variant='mvit_base_32x3', pretrained=True)
model = model.to(device)

# # DDP 파라미터 전용
# checkpoint_path = 'src/checkpoints/best_modelv2_aihub_nopadding_epoch_12.pth'

# checkpoint = torch.load(checkpoint_path, map_location=device)
# state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint


# new_state_dict = {}
# for key, value in state_dict.items():
#     if key.startswith("module."):
#         new_key = key[len("module."):] 
#         new_state_dict[new_key] = value
#     else:
#         new_state_dict[key] = value

# model.load_state_dict(new_state_dict)
model.eval()


test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_webcam_frames(frames_list, transform, max_frames=32, frame_size=(224, 224)):
    frames = []
    for frame in frames_list:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, frame_size)
        frame = Image.fromarray(frame)
        if transform:
            frame = transform(frame)
        frames.append(frame)

    if len(frames) < max_frames:
        frames.extend([frames[-1]] * (max_frames - len(frames)))

    frames = torch.stack(frames, dim=0)
    frames = frames.permute(1, 0, 2, 3)  
    return frames.unsqueeze(0).to(device)  

def inference_from_webcam(frames_list):
    frames = load_webcam_frames(frames_list[-32:], test_transforms)
    with torch.no_grad():
        outputs = model(frames)
        probabilities = nn.Softmax(dim=1)(outputs)
        _, preds = torch.max(outputs, 1)
    return preds.item(), probabilities


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


def process_and_infer(frames_list):
    while True:
        if len(frames_list) >= 32:  
            prediction, probabilities = inference_from_webcam(frames_list)
            frames_list.clear()  
            print(f"Predicted class: {class_names[prediction]}")

            
            for i, prob in enumerate(probabilities.squeeze()):
                percentage = prob.item() * 100
                print(f"{class_names[i]}: {percentage:.2f}%")


cap = cv2.VideoCapture(0) 
frames_list = []


infer_thread = threading.Thread(target=process_and_infer, args=(frames_list,))
infer_thread.start()

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_list.append(frame)  

        
        cv2.imshow('Webcam Action Recognition', frame)

        
        if cv2.waitKey(1) & 0xFF == 27:
            break


cap.release()
cv2.destroyAllWindows()







