import os
import glob
import random
import json
import numpy as np
import cv2
import pickle
from tqdm import tqdm
import time

with open("data/action/ucf5.pkl", "rb") as f:
    pkl_data = pickle.load(f)

print(pkl_data['split']['xsub_train'][0])

train_data = []
valid_data = []

label_dict = {'Archery': 0, 'BenchPress': 1, 'HandstandWalking': 2, 'PlayingGuitar': 3, 'SkateBoarding': 4}

for video_dir in pkl_data['split']['xsub_train']:
    train_data.append([video_dir, label_dict[video_dir.split("/")[-2]]])

with open('output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['path', 'label'])
    writer.writerows(train_data)