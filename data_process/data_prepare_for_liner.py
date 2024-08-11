import torch
import torch as t
import os
import os.path
import numpy as np
from torch.nn.utils.rnn import pad_sequence

target_s_p_1=t.load("image_predict_object_area")
num_target_p_1=t.load("image_predict_object_num")
image_tag=t.load("image_label")
k_list=t.load("image_file_name")
data_small=t.load("small_model_results")
target_num_big=t.load("image_object_num_big_model")
target_num_small=t.load("image_object_num_small_model")

train_data_x=[]
train_data_y=[]
max_num=18

for i in k_list:
    temp=[]
    a_small = t.from_numpy(np.array(data_small[i])[:, 0])
    mask = a_small.ge(0.5000)
    b_small = a_small[mask].numel()
    temp.append(b_small)
    temp.append(num_target_p_1[i])
    target_s_p_1[i].sort()
    for j in target_s_p_1[i]:
        temp.append(j)
    if len(target_s_p_1[i])<max_num:
        pad_num=max_num-len(target_s_p_1[i])
        for k in range(0,pad_num):
            temp.append(0)
    train_data_x.append(temp)
    train_data_y.append(image_tag[i])

x=torch.Tensor(train_data_x)
y=torch.Tensor(train_data_y)


torch.save(x,"data_for_discriminator")
torch.save(y,"data_label_for_discriminator")

target_num_big=t.load("image_object_num_big_model")
target_num_small=t.load("image_object_num_small_model")

k_list=t.load("image_file_name")
target_num_small_data=[]
target_num_big_data=[]
for i in k_list:
    target_num_small_data.append(target_num_small[i])
    target_num_big_data.append(target_num_big[i])
print(max(target_num_small_data))
print(max(target_num_big_data))
print(sum(target_num_small_data))
print(sum(target_num_big_data))
torch.save(target_num_big_data,"object_num_data_test_big_model")
torch.save(target_num_small_data,"object_num_data_test_small_model")
