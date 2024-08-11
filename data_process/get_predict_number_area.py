import torch as t
import numpy as np

data_small=t.load("small_model_results")
k_list=t.load("image_file_name")
image_area = t.load("image_size")

num_target_p_1={}
theshold=confidence_threshold
for k in k_list:
    data_small_1_20=t.from_numpy(np.array(data_small[k])[:,0])
    mask=data_small_1_20.ge(theshold/1000)
    num_target=data_small_1_20[mask].numel()
    if num_target==0:
        num_target=1
    num_target_p_1[k]=num_target

t.save(num_target_p_1,"image_predict_object_num")

target_s_p_1={}
for k in k_list:
    counter=0
    data_small_t=np.array(data_small[k])
    a=data_small_t.shape[0]
    for i in range(0,a):
        if data_small_t[i][0]>theshold/1000:
            s_temp=(data_small_t[i][3]-data_small_t[i][1])*(data_small_t[i][4]-data_small_t[i][2])
            s=s_temp.item()/image_area[k]
            target_s_p_1.setdefault(k,[]).append(s)
            counter+=1
    if counter==0:
        target_s_p_1.setdefault(k,[]).append(0)

t.save(target_s_p_1,"image_predict_object_area")