## DCSB(TMC'25+ICDCS'23)
---

## DISCRIMINATOR-TMC'25
# Data Process
1.To prepare the output results for both the small and large models, you can save the results in .txt files. Here's a step-by-step guide on how to structure and save the outputs:
Run both the small and large models on your dataset.
Ensure that each model's output (e.g., predictions, probabilities, bounding boxes) is captured.

2.Runing data_process/label_img.py, get_confidence_threshold.py, get_predict_number_area.py.

3.Runing data_prepare_for_liner.py

# Training 
Running discriminator.py

# Evaluation 
Running eval.py

## VIDEO
Selecting detection model and running detection_track_video.py

## Paper
https://ieeexplore.ieee.org/abstract/document/10705683

## Huawei-kubedge
Our algorithm has been integrated into the kubedge subproject Sedna.
Sedna:https://github.com/kubeedge/sedna/blob/main/examples/joint_inference/helmet_detection_inference

## Old_version_discriminator-ICDCS'23
## How to work
目标检测上云算法使用文档.doc  
## paper
https://ieeexplore.ieee.org/abstract/document/10272511
