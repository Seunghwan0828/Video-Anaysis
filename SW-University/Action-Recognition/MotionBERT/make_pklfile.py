import os
import glob
import random
import json
import numpy as np
import cv2
import pickle
from tqdm import tqdm
import time

random.seed(42)
np.random.seed(42)

pkl_data = {
    "split": {'xsub_train': [], 'xsub_val': [], 'xview_train': [], 'xview_val': []},
    "annotations": [],
        # {'frame_dir': [],
        # 'label': [],
        # 'img_shape': [],
        # 'original_shape': [],
        # 'total_frames': [],
        # 'keypoint': [],
        # 'keypoint_score': []
        # }
    }

video_dir = glob.glob("data/aihub/videos/*")
label_dict = {}

for n, vd in enumerate(video_dir):
    label_dict[os.path.basename(vd)] = n
    video_data = glob.glob(os.path.join(vd, "*.mp4"))
    random.shuffle(video_data)

    split_point = int(0.8 * len(video_data))
    train_data = video_data[:split_point]
    valid_data = video_data[split_point:]

    pkl_data['split']['xsub_train'] += train_data
    pkl_data['split']['xsub_val'] += valid_data

    for video_path in tqdm(train_data):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        print("Video:", video_path)
        command = f"python scripts/demo_inference.py --cfg configs/coco/hrnet/256x192_w32_lr1e-3.yaml --checkpoint pretrained_models/hrnet_w32_256x192.pth --video {video_path} --outdir examples/res"
        os.system(command)
        time.sleep(0.2)

        with open("examples/res/alphapose-results.json", "r") as f:
            json_data = json.load(f)
            
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
            'frame_dir': video_path,
            'label': n,
            'img_shape': (height, width),
            'original_shape': (height, width),
            'total_frames': total_frames,
            'keypoint': keypoint_data,
            'keypoint_score': keypoint_score_data}
        
        pkl_data['annotations'].append(pkl_dict)

    for video_path in tqdm(valid_data):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        print("Video:", video_path)
        command = f"python scripts/demo_inference.py --cfg configs/coco/hrnet/256x192_w32_lr1e-3.yaml --checkpoint pretrained_models/hrnet_w32_256x192.pth --video {video_path} --outdir examples/res"
        os.system(command)
        time.sleep(0.2)

        with open("examples/res/alphapose-results.json", "r") as f:
            json_data = json.load(f)

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
            'frame_dir': video_path,
            'label': n,
            'img_shape': (height, width),
            'original_shape': (height, width),
            'total_frames': total_frames,
            'keypoint': keypoint_data,
            'keypoint_score': keypoint_score_data
        }

        pkl_data['annotations'].append(pkl_dict)

    with open(f'ucf5{os.path.basename(vd)}.pkl', 'wb') as f:
        pickle.dump(pkl_data, f)
    print(f"save {os.path.basename(vd)}data")


# print("split dict keys:", list(pkl_data['split'].keys()))
# print()
# print("xsub_train data len:", len(pkl_data['split']['xsub_train']))
# print("xsub_train data 0:", pkl_data['split']['xsub_train'][0])
# print("xsub_val data len:", len(pkl_data['split']['xsub_val']))
# print("xsub_val data 0:", pkl_data['split']['xsub_val'][0])

# i = 0

# print("annotation dict keys:", list(pkl_data['annotations'][i].keys()), sep="\n", end='\n\n')
# print(len(pkl_data['annotations']))
# print("frame_dir:", pkl_data['annotations'][i]['frame_dir'])
# print(type(pkl_data['annotations'][i]['frame_dir']))
# print("label:", pkl_data['annotations'][i]['label']) # idx=n data label: idx % 60
# print(type(pkl_data['annotations'][i]['label']))
# print("img_shape:", pkl_data['annotations'][i]['img_shape'])
# print(type(pkl_data['annotations'][i]['img_shape']))
# print("original_shape:", pkl_data['annotations'][i]['original_shape'])
# print(type(pkl_data['annotations'][i]['original_shape']))
# print("total_frames:", pkl_data['annotations'][i]['total_frames'])
# print(type(pkl_data['annotations'][i]['total_frames']))
# print("keypoint shape:", pkl_data['annotations'][i]['keypoint'].shape)
# print(type(pkl_data['annotations'][i]['keypoint']))
# print(pkl_data['annotations'][i]['keypoint'][0][0][0])
# print("keypoint_score shape:", pkl_data['annotations'][i]['keypoint_score'].shape)
# print(type(pkl_data['annotations'][i]['keypoint_score']))
# print(pkl_data['annotations'][i]['keypoint_score'][0][0][0])