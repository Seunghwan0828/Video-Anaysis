import torch
import torch.optim as optim
import os
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pytorchvideo.losses.soft_target_cross_entropy import SoftTargetCrossEntropyLoss
from torchvision import transforms
from src.model import CustomMViTv2Model
from src.dataset import Ucf05Dataset
import wandb
import warnings
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore", category=UserWarning)

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = '172.17.0.2'
    os.environ['MASTER_PORT'] = '23456' 
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

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

# def custom_collate_fn(batch):
#     sequences = []
#     labels = []
    
#     for frames, label in batch:
#         sequences.append(frames)
#         labels.append(label)
    
#     sequences = torch.stack(sequences, dim=0)
#     labels = torch.tensor(labels, dtype=torch.long)

#     return sequences, labels

def train(rank, world_size):
    setup_ddp(rank, world_size)
    
    device = f"cuda:{rank}"

    batch_size_per_gpu = 32 // world_size
    num_workers = 8

    scaler = GradScaler()

    torch.backends.cudnn.benchmark = True

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    train_dataset = Ucf05Dataset(csv_file='data/aihub/annotations/train.csv', mode='train', transform=train_transforms)
    val_dataset = Ucf05Dataset(csv_file='data/aihub/annotations/val.csv', mode='val', transform=val_transforms)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_per_gpu, shuffle=False, collate_fn = custom_collate_fn, num_workers=num_workers, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_per_gpu, shuffle=False, collate_fn = custom_collate_fn, num_workers=num_workers, sampler=val_sampler)

    dataloaders = {'train': train_loader, 'val': val_loader}

    model = CustomMViTv2Model(num_classes=8, model_variant='mvit_base_32x3', pretrained=True)
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    
    for param in model.parameters():
        param.requires_grad = True

    criterion = SoftTargetCrossEntropyLoss(normalize_targets=False)
    optimizer = optim.AdamW(model.parameters(), lr=0.00001 * world_size, weight_decay=1e-4) 
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6) 
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)


    best_acc = 0.0
    epochs_without_improvement = 0
    save_dir = "src/checkpoints/aihub"
    # patience = 5
    num_epochs = 30

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    best_model_wts = None

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                train_sampler.set_epoch(epoch)
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            phase_loader = tqdm(dataloaders[phase], desc=f"{phase} Epoch {epoch+1}/{num_epochs}")

            for inputs, labels in phase_loader:
                inputs = inputs.to(device)
                labels = labels.to(device) 

                optimizer.zero_grad()

                with autocast():
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            total_loss = torch.tensor(running_loss).to(device)
            total_corrects = torch.tensor(running_corrects).to(device)
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_corrects, op=dist.ReduceOp.SUM)

            epoch_loss = total_loss.item() / len(dataloaders[phase].dataset)
            epoch_acc = total_corrects.item() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            

            if phase == 'train':
                scheduler.step()

            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    epochs_without_improvement = 0
                    checkpoint_path = os.path.join(save_dir, f'best_modelv2_aihub_final_epoch_{epoch}.pth')
                    torch.save(best_model_wts, checkpoint_path)
                    print(f'Saved best model checkpoint: {checkpoint_path}')
                else:
                    epochs_without_improvement += 1

       

    print(f'Best val Acc: {best_acc:.4f}')

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    cleanup_ddp()

if __name__ == "__main__":
    mp.set_start_method('fork', force=True)
    world_size = 4
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    
