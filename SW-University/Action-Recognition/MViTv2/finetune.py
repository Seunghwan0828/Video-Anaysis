
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import torch.hub
from functools import partial
from pytorchvideo.losses.soft_target_cross_entropy import SoftTargetCrossEntropyLoss
import os
import torch
import torch.utils.data
from torchvision import transforms
# from slowfast.datasets.transform import *
from src.model import CustomMViTv2Model
from src.dataset import Ucf05Dataset
import wandb
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"



def custom_collate_fn(batch):
    max_len = max([x[0].shape[0] for x in batch])
    padded_sequences = []
    labels = []
    for frames, label in batch:
        pad_len = max_len - frames.shape[0]
        if pad_len > 0:
            padding = [frames[-1].unsqueeze(0).repeat(pad_len, 1, 1, 1)]
            frames = torch.cat([frames] + padding, dim=0)
        padded_sequences.append(frames)
        labels.append(label)
    padded_sequences = torch.stack(padded_sequences, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_sequences, labels

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomCrop(224),   
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    # transforms.RandomRotation(degrees=(0, 90)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = Ucf05Dataset(csv_file='data/aihub/annotations/train.csv', mode='train', transform=train_transforms)
val_dataset = Ucf05Dataset(csv_file='data/aihub/annotations/val.csv', mode='val', transform=train_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

model = CustomMViTv2Model(num_classes=8, model_variant='mvit_base_32x3', pretrained=True)
model = model.to(device)

criterion = SoftTargetCrossEntropyLoss(normalize_targets=False)
optimizer = optim.AdamW(model.parameters(), lr=0.0001)



def train_model(model, dataloaders, criterion, optimizer, num_epochs=10, save_dir="src/checkpoints/aihub", patience=5):
    best_acc = 0.0
    epochs_without_improvement = 0
    model = model.to(device)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    best_model_wts = None

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            phase_loader = tqdm(dataloaders[phase], desc=f"{phase} Epoch {epoch+1}/{num_epochs}")

            for inputs, labels in phase_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    epochs_without_improvement = 0
                    checkpoint_path = os.path.join(save_dir, f'best_modelv2_aihub_epoch_{epoch}.pth')
                    torch.save(best_model_wts, checkpoint_path)
                    print(f'Saved best model checkpoint: {checkpoint_path}')
                else:
                    epochs_without_improvement += 1

        # if best_acc == 1.0 or epochs_without_improvement >= patience:
        if epochs_without_improvement >= patience:
            print(f'Early stopping triggered. Best val Acc: {best_acc:.4f}')
            break

    print(f'Best val Acc: {best_acc:.4f}')

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    return model


dataloaders = {'train': train_loader, 'val': val_loader}

model = train_model(model, dataloaders, criterion, optimizer, num_epochs=30, patience=5)


