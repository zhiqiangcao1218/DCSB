#-------------------------------获取决策器阈值---------------------------------------------------------------------#
from SemanticInformationFilter import SemanticInformationFilter as Sif
data_big="F:\\project\\yolov4_voc2007.pkl"   #训练集在大模型的检测结果路径
data_small="F:\\project\\mobilev1_yolov4_voc2007.pkl"  #训练集在小模型的检测结果路径
path_annotations = "./Annotations"    #测试集annotation路径

sif_discriminator = Sif(data_big_path=data_big,data_small_path=data_small,
                        path_annotations=path_annotations)  #实例化
sif_discriminator.get_threshold()  #生成决策器的阈值
#-----------------使用决策器阈值-------------------------------------------------------------------------------------#
# result_small_single_picture_path="single-picture-detection-small-path" #待检测的单张图片在小模型的检测结果路径
# image_type = Sif.discriminator(threshold_confidence=198,threshold_area=119,threshold_number=4,
#                                result_small_single_picture_path=result_small_single_picture_path)   #调用决策器判断
# print(image_type)
