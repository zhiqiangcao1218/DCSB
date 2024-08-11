import torch as t
import numpy as np

data_big=t.load("big_model_results")
data_small=t.load("small_model_results")
image_target_s=t.load("ground_truth_area")
k_list=t.load("image_file_name")
SUM_big=0
SUM_small=0
target_num_yolov4={}
target_num_mobilev1_yolov4={}
image_tag={}
image_index_n=[]
print(len(k_list))
for j in k_list:
    a_big=t.from_numpy(np.array(data_big[j])[:,0])
    mask=a_big.ge(0.5000)
    b_big=a_big[mask].numel()
    if b_big<=len(image_target_s[j]):
        target_num_big=b_big
    else:
        target_num_big=len(image_target_s[j])
    target_num_yolov4[j]=target_num_big
    SUM_big+=target_num_big
    a_small=t.from_numpy(np.array(data_small[j])[:,0])
    mask=a_small.ge(0.5000)
    b_small=a_small[mask].numel()
    if b_small <= len(image_target_s[j]):
        target_num_small = b_small
    else:
        target_num_small = len(image_target_s[j])
    target_num_small=b_small
    SUM_small+=target_num_small
    target_num_mobilev1_yolov4[j]=target_num_small
    miss=target_num_big-target_num_small
    miss_thread=1
    if miss>=miss_thread:
        label=1
        image_tag[j]=label
        image_index_n.append(j)
    else:
        label=0
        image_tag[j]=label

t.save(image_tag,'image_label')
t.save(target_num_yolov4,'image_object_num_big_model')
t.save(target_num_mobilev1_yolov4,'image_object_num_samll_model')

