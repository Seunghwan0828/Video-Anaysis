import os
import numpy as np
import time
import sys
import argparse
import errno
from collections import OrderedDict
import tensorboardX
from tqdm import tqdm
import random
import json
import pickle
import time
import cv2
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.model.loss import *
from lib.data.dataset_action import NTURGBD
from lib.model.model_action import ActionNet

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-v', '--video', default='example_video.mp4', type=str, metavar='FILENAME', help='infer video file')
    opts = parser.parse_args()
    return opts

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        new_state_dict[new_key] = v
    return new_state_dict

def keypoints_estimation(opts):
    infer_video_path = opts.video
    print("Video:", infer_video_path)

    # command = []
    # command.append('cd ../AlphaPose')
    # command.append(f"python scripts/demo_inference.py --cfg configs/coco/hrnet/256x192_w32_lr1e-3.yaml --checkpoint pretrained_models/hrnet_w32_256x192.pth --video ../MotionBERT{infer_video_path} --outdir ../MotionBERT/examples/action")
    # command.append('cd ../MotionBERT')
    # for cmd in command:
    #     print("current command:", cmd)
    #     os.system(cmd)
    #     time.sleep(0.1)

    cap = cv2.VideoCapture(infer_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    with open(f"{infer_video_path.replace('mp4', 'json')}", "rb") as f:
        json_data = json.load(f)

    pkl_data = {
        "split": {'xsub_train': [], 'xsub_val': [], 'xview_train': [], 'xview_val': []},
        "annotations": [],
    }

    total_frames = len(json_data)
    keypoint_data = []
    keypoint_score_data = []

    for frame in json_data:
        keypoint = frame['keypoints']
        for i, v in enumerate(keypoint):
            if i % 3 == 0:
                x = round(v, 1)
            elif i % 3 == 1:
                y = round(v, 1)
            else:
                keypoint_data.append((x, y))
                keypoint_score_data.append(round(v, 3))

    keypoint_data = np.array(keypoint_data).reshape(1, total_frames, 17, 2)
    keypoint_score_data = np.array(keypoint_score_data).reshape(1, total_frames, 17)

    pkl_dict = {
        'frame_dir': infer_video_path,
        'label': 3,
        'img_shape': (height, width),
        'original_shape': (height, width),
        'total_frames': total_frames,
        'keypoint': keypoint_data,
        'keypoint_score': keypoint_score_data
    }
    pkl_data['split']['xsub_val'].append(infer_video_path)
    pkl_data['annotations'].append(pkl_dict)

    with open(f"{infer_video_path.replace('mp4', 'pkl')}", "wb") as f:
        pickle.dump(pkl_data, f)
        print('Save infer pkl data')


def infer_with_config(args, opts):
    start_time = time.time()

    print(opts)
    keypoints_estimation(opts=opts)
    labels = ['A1. Walking', 'A2. Situp', 'A3. Running', 'A4. Fallingdown', 'A5. Jumping', 'A6. Climbingstairs', 'A7. Sitting', 'A8. Pushup']

    model_backbone = load_backbone(args)
    model = ActionNet(
        backbone=model_backbone,
        dim_rep=args.dim_rep,
        num_classes=args.action_classes,
        dropout_ratio=args.dropout_ratio,
        version=args.model_version,
        hidden_dim=args.hidden_dim,
        num_joints=args.num_joints
    )
    chk_filename = os.path.join(opts.pretrained, "best_epoch.bin")
    print('Loading pretrained checkpoint', chk_filename)
    pretrained_checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    state_dict = remove_module_prefix(pretrained_checkpoint['model'])
    model.load_state_dict(state_dict, strict=True)
    model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss().cuda()

    inferloader_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 16,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True
    }
    infer_data = NTURGBD(data_path=f'{opts.video.replace(".mp4", "")}.pkl', data_split=args.data_split+'_val',
        n_frames=args.clip_len, random_move=False, scale_range=args.scale_range_test)
    infer_loader = DataLoader(infer_data, **inferloader_params)

    input_infer_data, gt = next(iter(infer_loader))
    input_infer_data = input_infer_data.cuda()
    gt = gt.cuda()
    
    model.eval()
    infer_time = time.time()
    with torch.no_grad():
        output = model(input_infer_data).squeeze()

    softmax = nn.Softmax(dim=0)
    probabilities = softmax(output)

    end_time = time.time()

    print(f"\nTotal Time: {round(end_time-start_time, 3)}")
    print(f"Inference Time: {round(end_time-infer_time, 3)}")
    print('--------Infer Result--------')
    for i in range(len(labels)):
        print(f"{labels[i]}: {probabilities[i]*100:0.2f} %")
    print('----------------------------')

    

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    infer_with_config(args, opts)