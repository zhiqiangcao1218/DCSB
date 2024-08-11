import numpy as np
import cv2
import time
import os
import torch
from numpy import *


def compute_iou(rec1, rec2):
    """
    computing IoU
    rec1: (x0, y0, x1, y1)
    rec2: (x0, y0, x1, y1)
    :return: scala value of IoU
    """
    # computing area of each rectangle
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
    #print(top_line, left_line, right_line, bottom_line)

    # judge if there is an intersect area
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def draw_box(points,img):
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box_x = [*box[:, 0]]
    box_y = [*box[:, 1]]
    x_min = min(box_x)
    y_min = min(box_y)
    x_max = max(box_x)
    y_max = max(box_y)
    cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
    return [x_min, y_min, x_max, y_max]


def mask_data(mask_use,mask):
    mask_use[mask[0]:mask[2],mask[1]:mask[3]]=255


def NCC(old_frame,new_frame):

    H, W, C = old_frame.shape
    print(H, W, C)

    # Read templete image
    Ht, Wt, Ct = new_frame.shape
    print(Ht, Wt, Ct)

    v = -1
    for y in range(H - Ht):
        for x in range(W - Wt):
            _v = np.sum(old_frame[y:y + Ht, x:x + Wt] * new_frame)
            _v /= (np.sqrt(np.sum(old_frame[y:y + Ht, x:x + Wt] ** 2)) * np.sqrt(np.sum(new_frame ** 2)))
            if _v > v:
                v = _v
    return v

def pearson(img1,img2):
    v=np.mean(np.multiply((img1-np.mean(img1)),(img2-np.mean(img2))))/(np.std(img1)*np.std(img2))
    return v



cap = cv2.VideoCapture('video_path')

feature_params = dict(maxCorners=100,
                      qualityLevel=0.2,
                      minDistance=1,
                      blockSize=1)


lk_params = dict(winSize=(15,15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)


mask_use_black= np.zeros(old_gray.shape,np.uint8)
mask_black=[96,563,437,854]# Helmet
mask_use_black[mask_black[0]:mask_black[2],mask_black[1]:mask_black[3]]=255
detector = cv2.ORB_create()
p0_black=detector.detect(old_gray,mask=mask_use_black)
p0_BLACK= np.empty([len(p0_black),1,2],dtype=np.float32)
def points_data(p0,p0_):
    for i in range(len(p0)):
        a = int(p0[i].pt[0])
        b = int(p0[i].pt[1])
        p0_[i][0] = [a, b]

points_data(p0_black,p0_BLACK)

frame_num = 1
frame_box={}
track_gt=torch.load("ground_truth")

# model="fixed"
# model="speed"
model="dynamic"
# model="context"

frame_old=1
sample_init=25
climb=10
climb_sign=0
climb_up=1.2
climb_dowm=0.5
count_dete=0
context_speed=20
period=[1,2,3]
sample_init=2
frame_init=1
temp_x_10=[]
temp_y_10=[]
temp=[]
speed_ratio=10
speed_threshold=3
while True:
    temp_x = []
    temp_y = []
    frame_num = frame_num + 1
    ret, frame = cap.read()
    if frame is None:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1_black, st_black, err_black = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0_BLACK, None, **lk_params)
    good_old_black=p0_BLACK[st_black==1]
    good_new_black= p1_black[st_black == 1]
    box_track_black = draw_box(img=frame, points=good_new_black)
    frame_box.setdefault(frame_num,[]).append(box_track_black)
    cv2.imshow('frame', frame)
    fileindex = str(frame_num)
    k = cv2.waitKey(30)  # & 0xff
    if k == 27:
        break
    old_gray = frame_gray.copy()
    p0_BLACK = good_new_black.reshape(-1, 1, 2)

    if model=="context":
        if (frame_num - frame_init) % (context_speed / sample_init) == 0:
            count_dete = count_dete + 1
            rec1 = box_track_black
            rec2 = track_gt[str(frame_num + 15844)]
            iou_temp = compute_iou(rec1, rec2)
            temp.append(iou_temp)
            mask_use_black_6 = np.zeros(old_gray.shape, np.uint8)
            mask_black_6 = [track_gt[str(frame_num + 15844)][1], track_gt[str(frame_num + 15844)][0],
                            track_gt[str(frame_num + 15844)][3], track_gt[str(frame_num + 15844)][2]]
            mask_data(mask_use_black_6, mask_black_6)
            p0_black = detector.detect(old_gray, mask=mask_use_black_6)
            p0_BLACK = np.empty([len(p0_black), 1, 2], dtype=np.float32)
            points_data(p0_black, p0_BLACK)
        if (frame_num - 1) % context_speed == 0:
            iou_average=sum(temp)/len(temp)
            if iou_average>=0.5:
                sample_init = period[0]
            elif 0.3<iou_average<=0.5:
                sample_init = period[1]
            elif iou_average<=0.3:
                sample_init = period[2]
            temp=[]

    if model == "speed":
        for i in range(0, len(good_new_black)):
            temp_x.append(good_new_black[i][0] - good_old_black[i][0])
            temp_y.append(good_new_black[i][1] - good_old_black[i][1])
        temp_x_10.append(sum(temp_x) / len(good_new_black))
        temp_y_10.append(sum(temp_y) / len(good_new_black))
        if (frame_num - 1) % speed_ratio == 0:
            speed_x = abs(sum(temp_x_10) / len(temp_x_10))
            speed_y = abs(sum(temp_y_10) / len(temp_y_10))
            print("speed_x:", speed_x)
            print("speed_y:", speed_y)
            temp_x_10 = []
            temp_y_10 = []
            if speed_x > speed_threshold or speed_y > speed_threshold:
                count_dete = count_dete + 1
                mask_use_black_6 = np.zeros(old_gray.shape, np.uint8)
                mask_black_6 = [track_gt[str(frame_num + 15844)][1], track_gt[str(frame_num + 15844)][0],
                                track_gt[str(frame_num + 15844)][3], track_gt[str(frame_num + 15844)][2]]
                mask_data(mask_use_black_6, mask_black_6)
                p0_black = detector.detect(old_gray, mask=mask_use_black_6)
                p0_BLACK = np.empty([len(p0_black), 1, 2], dtype=np.float32)
                points_data(p0_black, p0_BLACK)

    if model=="fixed":

        if (frame_num - 1) % sample_init == 0:
            count_dete = count_dete + 1
            mask_use_black_6 = np.zeros(old_gray.shape, np.uint8)
            mask_black_6 = [track_gt[str(frame_num + 15844)][1], track_gt[str(frame_num + 15844)][0],
                            track_gt[str(frame_num + 15844)][3], track_gt[str(frame_num + 15844)][2]]
            mask_data(mask_use_black_6, mask_black_6)
            p0_black = detector.detect(old_gray, mask=mask_use_black_6)
            p0_BLACK = np.empty([len(p0_black), 1, 2], dtype=np.float32)
            points_data(p0_black, p0_BLACK)

    elif model=="dynamic":
        if frame_num - frame_old == sample_init:
            count_dete = count_dete + 1
            print(frame_old, frame_num, climb_sign, climb, climb_up, climb_dowm)
            rec1 = box_track_black
            rec2 = track_gt[str(frame_num + 15844)]
            iou_temp = compute_iou(rec1, rec2)
            if climb_sign == 0:
                if iou_temp >= 0.5:
                    climb_sign_new = 1
                    sample_init = sample_init + climb
                elif iou_temp < 0.5:
                    climb_sign_new = 2
                    sample_init = sample_init - climb
            elif climb_sign != 0:
                if iou_temp >= 0.5:
                    climb_sign_new = 1
                elif iou_temp < 0.5:
                    climb_sign_new = 2
                if climb_sign_new == climb_sign:
                    climb = climb * climb_up
                elif climb_sign_new != climb_sign:
                    climb = climb * climb_dowm

                if climb_sign_new == 1:
                    sample_init = sample_init + climb
                elif climb_sign_new == 2:
                    sample_init = sample_init - climb
            frame_old = frame_num
            climb_sign = climb_sign_new
            print(frame_old, frame_num, climb_sign, climb, climb_up, climb_dowm)
            mask_use_black_6 = np.zeros(old_gray.shape, np.uint8)
            mask_black_6 = [track_gt[str(frame_num + 15844)][1], track_gt[str(frame_num + 15844)][0],
                            track_gt[str(frame_num + 15844)][3], track_gt[str(frame_num + 15844)][2]]
            mask_data(mask_use_black_6, mask_black_6)
            p0_black = detector.detect(old_gray, mask=mask_use_black_6)
            p0_BLACK = np.empty([len(p0_black), 1, 2], dtype=np.float32)
            points_data(p0_black, p0_BLACK)

torch.save(frame_box,"tracking_results")
print(count_dete)
cv2.destroyAllWindows()
cap.release()
