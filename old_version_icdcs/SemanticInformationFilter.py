import os.path
import os
from xml.etree import ElementTree as ET
import torch as t
import numpy as np
from tqdm import tqdm


class  SemanticInformationFilter(object):
    def __init__(self,data_big_path, data_small_path,path_annotations):
        self.data_big_path = data_big_path
        self.data_small_path = data_small_path
        self.path_annotations = path_annotations
        self.default_range_threshold = 15                                   ####number threshold search range
        self.default_loss_function = 1                                       #####loss function selected 1/2/3


    def get_threshold(self):

        path = self.data_small_path
        data_small = {}
        dir_list = os.listdir(path)
        for i in dir_list:
            file_path = os.path.join(path, i)
            file_data = open(file_path, 'r').readlines()
            for row in file_data:
                tmp_list = row.split()
                tmp_list[-1] = tmp_list[-1].replace('\n', '')
                data_small.setdefault(i.split(".")[0], []).append(list(map(float, tmp_list[1:])))

        path = self.data_big_path
        data_big = {}
        dir_list = os.listdir(path)
        for i in dir_list:
            file_path = os.path.join(path, i)
            file_data = open(file_path, 'r').readlines()
            for row in file_data:
                tmp_list = row.split()
                tmp_list[-1] = tmp_list[-1].replace('\n', '')
                data_big.setdefault(i.split(".")[0], []).append(list(map(float, tmp_list[1:])))

        path = self.path_annotations
        k_list = []
        image_target_s = {}
        target_number_big = {}
        target_number_small = {}
        num_target_p = {}
        area_target_p = {}
        threshold_discriminator = {}
        Differ = {}
        for i in data_big.keys():
            k_list.append(i)
        file = k_list
        for f in file:
            file_name = f + ".xml"
            path_test = os.path.join(path, file_name)
            tree = ET.parse(path_test)
            root = tree.getroot()
            size = root.find("size")
            Width = int(size.find("width").text)
            Height = int(size.find("height").text)
            ground_s = Width * Height
            Nodes = root.findall("object")
            for Node in Nodes:
                bndbox = Node.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                xmax = int(bndbox.find("xmax").text)
                ymin = int(bndbox.find("ymin").text)
                ymax = int(bndbox.find("ymax").text)
                width = xmax - xmin
                height = ymax - ymin
                s = width * height / ground_s
                image_target_s.setdefault(f, []).append(s)

        for j in k_list:

            a_big = t.from_numpy(np.array(data_big[j])[:, 0])
            mask = a_big.ge(0.5000)
            target_number_big[j] =a_big[mask].numel()
            if target_number_big[j] > len(image_target_s[j]):  # 序号为j的图片的大网络检测出目标数量
               target_number_big[j] = len(image_target_s[j])

            a_small = t.from_numpy(np.array(data_small[j])[:, 0])
            mask = a_small.ge(0.5000)
            target_number_small[j] = a_small[mask].numel()


        for i in range(1, 501):
            differ = 0
            for k in k_list:
                num_target = 0
                image_target_num = len(image_target_s[k])
                #data_small_1_20 = data_small[k][1:, :, 0]
                data_small_1_20 =t.from_numpy(np.array(data_small[k])[:, 0])
                temp = data_small_1_20 >= i / 1000
                num_target += temp.sum().item()
                if num_target == 0:
                    num_target = 1
                if self.default_loss_function ==1:
                    differ += num_target - image_target_num         ####loss function select
                if self.default_loss_function ==2:
                    differ += abs(num_target - image_target_num)
                if self.default_loss_function ==3:
                    differ+=abs((num_target-image_target_num)/image_target_num)
            Differ[i] = abs(differ)
        threshold_confidence = min(Differ, key=Differ.get)
        print("GET THRESHOLD-CONFIDENCE!!!")

        for k in k_list:
            num_target = 0
            data_small_1_20 =t.from_numpy(np.array(data_small[k])[:, 0])
            temp = data_small_1_20 >= threshold_confidence / 1000
            num_target += temp.sum().item()
            if num_target == 0:
                num_target = 1
            num_target_p[k] = num_target

        for k in k_list:
            counter = 0
            data_small_t = np.array(data_small[k])
            a = data_small_t.shape[0]
            for i in range(0, a):
                if data_small_t[i][0] > threshold_confidence / 1000:
                    s_temp = (data_small_t[i][3] - data_small_t[i][1]) * (data_small_t[i][4] - data_small_t[i][2])
                    s = s_temp.item()
                    area_target_p.setdefault(k, []).append(s)
                    counter += 1
            if counter == 0:
                area_target_p.setdefault(k, []).append(0)


        for j in tqdm(range(1, self.default_range_threshold)):
            for i in tqdm(range(1, 1001)):
                num_upload = 0
                SUM = 0
                for k in k_list:
                    temp_sum = 0
                    image_target_num = num_target_p[k]
                    area_target_p[k].sort()
                    image_target_minimun = area_target_p[k][0]
                    data_small_1_20 = t.from_numpy(np.array(data_small[k])[:, 0])
                    temp = data_small_1_20 >= 0.5
                    temp_sum += temp.sum().item()
                    if temp_sum == 0:
                        num_upload += 1
                        SUM += target_number_big[k]
                    else:
                        if temp_sum == image_target_num:
                            SUM += target_number_small[k]
                        else:
                            if image_target_minimun >= i / 1000 and image_target_num < j:
                                SUM += target_number_small[k]
                            else:
                                num_upload += 1
                                SUM += target_number_big[k]
                target_percentage = SUM / sum(target_number_big.values())
                cloud_percentage = num_upload / len(k_list)
                print(j,i,cloud_percentage)
                if 0.39 < cloud_percentage < 0.41:
                    threshold_discriminator[str(i) + "+" + str(j)] = target_percentage
        threshold_area_number = max(threshold_discriminator, key=threshold_discriminator.get)
        threshold_area = int(threshold_area_number.split("+")[0])
        threshold_number = int(threshold_area_number.split("+")[-1])
        threshold = {"threshold_confidence":threshold_confidence,"threshold_area":threshold_area,"threshold_number":threshold_number}
        f=open("threshold_list.txt",mode="w")
        f.write(str(threshold))
        f.close()
        print("GET THREE THRESHOLDS!!!")

    @classmethod
    def discriminator(cls,result_small_single_picture_path,threshold_confidence,threshold_area,threshold_number):


        path = result_small_single_picture_path
        result_small_single_picture = {}
        dir_list = os.listdir(path)
        for i in dir_list:
            file_path = os.path.join(path, i)
            file_data = open(file_path, 'r').readlines()
            for row in file_data:
                tmp_list = row.split()
                tmp_list[-1] = tmp_list[-1].replace('\n', '')
                result_small_single_picture.setdefault(i.split(".")[0], []).append(list(map(float, tmp_list[1:])))

        data_small_single_picture = result_small_single_picture
        threshold_confidence = threshold_confidence
        threshold_area = threshold_area
        threshold_number = threshold_number
        num_target = 0
        data_small_1_20 =t.from_numpy(np.array(data_small_single_picture)[:, 0])
        temp = data_small_1_20 >= threshold_confidence / 1000
        num_target += temp.sum().item()
        if num_target == 0:
            num_target = 1
        num_target_p_1 = num_target
        ###-----------------------------------###  预测最小目标的面积占比
        counter = 0
        target_s_p_1 = []
        data_small_t = np.array(data_small_single_picture)
        a = data_small_t.shape[0]
        for i in range(0, a):
            if data_small_t[i][0] > threshold_confidence / 1000:
                s_temp = (data_small_t[i][3] - data_small_t[i][1]) * (data_small_t[i][4] - data_small_t[i][2])
                s = s_temp.item()
                target_s_p_1.append(s)
                counter += 1
        if counter == 0:
            target_s_p_1.append(0)

        ###-----------------------------------###  难例判断
        temp_sum = 0
        image_target_num = num_target_p_1
        target_s_p_1.sort()
        image_target_minimum = target_s_p_1[0]
        data_small_1_20 = t.from_numpy(np.array(data_small_single_picture)[:, 0])
        temp = data_small_1_20 >= 0.5
        temp_sum += temp.sum().item()
        if temp_sum == 0:
            image_type = True
        else:
            if temp_sum == image_target_num:
                image_type = False
            else:
                if image_target_minimum >= threshold_area / 1000 and image_target_num < threshold_number:
                    image_type = False
                else:
                    image_type = True
        return image_type               #####  true-->upload  false-->local




