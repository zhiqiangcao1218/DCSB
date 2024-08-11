from discriminator import Discriminator
import torch
import numpy as np
import time
def metric(x,y,target_num_big_data,target_num_small_data):
    TP=0
    FP=0
    FN=0
    TN=0
    target_num=0
    img_index=[]
    count=0
    for i in range(0,len(y)):
        results=x[i]
        if results==1 and y[i]==1:
            TP=TP+1
            count=count+1
        if results==1 and y[i]==0:
            FP=FP+1
            count=count+1
        if results==0 and y[i]==1:
            FN=FN+1
        if results==0 and y[i]==0:
            TN=TN+1
        if results==0:
            if target_num_small_data[i]<=1:
                target_num=target_num+target_num_small_data[i]
            if target_num_small_data[i]>1:
                target_num=target_num+1
        if results == 1:
            if target_num_big_data[i]<=1:
                target_num = target_num + target_num_big_data[i]
            if target_num_big_data[i]>1:
                target_num = target_num + 1
    print("TP",TP)
    print("FP",FP)
    print("FN",FN)
    print("TN",TN)
    accuracy=(TN+TP)/len(y)
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    upload=(TP+FP)/len(y)
    print("upload:",upload)
    print("accuracy %.4f',precision %.4f',recall %.4f'" %(accuracy,precision,recall))
    print(target_num)
    print("target_percentage:",target_num/sum(target_num_big_data))


def evaluate_acc(x,y,net,target_num_big_data,target_num_small_data):
    net.eval()
    results_list=[]
    out = net(x)
    correct =(out.ge(0.5)==y).sum().item()
    result=out.ge(0.5)
    for i in result:
        if i ==True:
            results_list.append(1)
        else:
            results_list.append(0)
    n = y.shape[0]
    metric(x=results_list,y=y,target_num_big_data=target_num_big_data,target_num_small_data=target_num_small_data)
    return correct/n


net=Discriminator()
test_x=torch.load("small_model_output_results_test_data")
test_y=torch.load("small_model_output_results_label_test_data")
net.load_state_dict(torch.load("model_weight_path"))
target_num_big_data=torch.load("image_object_num_big_model")
target_num_small_data=torch.load("image_object_num_small_model")
train_acc=evaluate_acc(test_x,test_y,net,target_num_big_data=target_num_big_data,target_num_small_data=target_num_small_data)
