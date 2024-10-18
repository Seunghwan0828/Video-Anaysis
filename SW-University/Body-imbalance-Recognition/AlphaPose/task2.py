import pickle
import numpy as np

pkl_list = ['ucf5Archery.pkl', 'ucf5BenchPress.pkl', 'ucf5HandstandWalking.pkl', 'ucf5PlayingGuitar.pkl', 'ucf5SkateBoarding.pkl']
# pkl_data = {
#     'split': {
#         'xsub_train': [],
#         'xsub_val': [],
#         'xview_train': [],
#         'xview_val': [],
#     },
#     'annotations': []
# }

# for i in range(5):
#     with open(pkl_list[i], 'rb') as f:
#         pkl_list_data = pickle.load(f)
    
#     if len(pkl_list_data['split']['xsub_train']) > len(pkl_data['split']['xsub_train']):
#         pkl_data['split']['xsub_train'] = pkl_list_data['split']['xsub_train']
#     if len(pkl_list_data['split']['xsub_val']) > len(pkl_data['split']['xsub_val']):
#         pkl_data['split']['xsub_val'] = pkl_list_data['split']['xsub_val']

#     pkl_data['annotations'] += pkl_list_data['annotations']

# with open(f'ucf5.pkl', 'wb') as f:
#     pickle.dump(pkl_data, f)
# print(f"save {os.path.basename(vd)}data")\

with open(pkl_list[4], "rb") as f:
    pkl_data = pickle.load(f)

print("split dict keys:", list(pkl_data['split'].keys()))
print()
print("xsub_train data len:", len(pkl_data['split']['xsub_train']))
print("xsub_train data 0:", pkl_data['split']['xsub_train'][0])
print("xsub_val data len:", len(pkl_data['split']['xsub_val']))
print("xsub_val data 0:", pkl_data['split']['xsub_val'][0])

i = 0

print("annotation dict keys:", list(pkl_data['annotations'][i].keys()), sep="\n", end='\n\n')
print(len(pkl_data['annotations']))
print("frame_dir:", pkl_data['annotations'][i]['frame_dir'])
print(type(pkl_data['annotations'][i]['frame_dir']))
print("label:", pkl_data['annotations'][i]['label']) # idx=n data label: idx % 60
print(type(pkl_data['annotations'][i]['label']))
print("img_shape:", pkl_data['annotations'][i]['img_shape'])
print(type(pkl_data['annotations'][i]['img_shape']))
print("original_shape:", pkl_data['annotations'][i]['original_shape'])
print(type(pkl_data['annotations'][i]['original_shape']))
print("total_frames:", pkl_data['annotations'][i]['total_frames'])
print(type(pkl_data['annotations'][i]['total_frames']))
print("keypoint shape:", pkl_data['annotations'][i]['keypoint'].shape)
print(type(pkl_data['annotations'][i]['keypoint']))
print(pkl_data['annotations'][i]['keypoint'][0][0][0])
print("keypoint_score shape:", pkl_data['annotations'][i]['keypoint_score'].shape)
print(type(pkl_data['annotations'][i]['keypoint_score']))
print(pkl_data['annotations'][i]['keypoint_score'][0][0][0])