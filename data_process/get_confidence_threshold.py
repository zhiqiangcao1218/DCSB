import torch as t
import numpy as np
from tqdm import tqdm

data_small=t.load("small_model_results")
k_list=t.load("image_file_name")
image_target_s=t.load("ground_truth_area")


l_differ_1={}
differ_min=40000
l_min=0
for l in tqdm(range(1,501)):
    differ=0
    for k in k_list:
        num_target=0
        image_target_num=len(image_target_s[k])
        data_small_1_20=t.from_numpy(np.array(data_small[k])[:,0])
        temp=data_small_1_20>=l/1000
        num_target+=temp.sum().item()
        if num_target==0:
            num_target=1
        differ+=num_target-image_target_num
    if abs(differ)<=differ_min:
        differ_min=differ
        l_min=l
    l_differ_1[l]=differ
print(str(l_differ_1))
print("differ_min:",differ_min,"l_min",l_min)