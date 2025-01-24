# Underwater-Species-Detection-Using-YOLOv9-with-self---attention-mechanism
Underwater Species Detection Using YOLOv9 employs a self-attention mechanism for precise marine species identification. It enhances focus on key details, ensuring accuracy in low visibility and complex conditions. Designed for real-time use, it supports marine research, biodiversity studies, and conservation efforts.
!nvidia-smi
 Op
import os
HOME = os.getcwd()
print(HOME)
!git clone https://github.com/SkalskiP/yolov9.git
%cd yolov9
!pip install -r requirements.txt -q
!pip install -q roboflow
op
!ls -la {HOME}/weights
op
%cd {HOME}/yolov9
import roboflow
roboflow.login()
rf = roboflow.Roboflow()
project = rf.workspace("datasets-5ubnn").project("sea-ag1ue")
version = project.version(1)
dataset = version.download("yolov9")
import shutil
import os
# Define the source and destination base directories
source_base_dir = '/content/yolov9/yolov9/Sea-1'
destination_base_dir = '/content/yolov9/yolov9'
# Define the subdirectories to move (excluding data.yaml)
subdirectories = [
    "valid/images", "valid/labels",
    "test/images", "test/labels",
    "train/images", "train/labels"
    ]
# Iterate over each subdirectory and move its contents to the new destination
for subdirectory in subdirectories:
    source_path = os.path.join(source_base_dir, subdirectory)
    destination_path = os.path.join(destination_base_dir, subdirectory)
# If it's a file, move it directly; otherwise, move the entire directory
    if os.path.isfile(source_path):
        shutil.move(source_path, destination_base_dir)
        print(f'Moved file: {source_path}')
    elif os.path.isdir(source_path):
        os.makedirs(destination_path, exist_ok=True)  # Create the destination directory if it doesn't exist
        for item in os.listdir(source_path):
            shutil.move(os.path.join(source_path, item), destination_path)
        print(f'Moved directory: {source_path} to {destination_path}')
!cat /content/yolov9/Sea-1/data.yaml
%cd {HOME}/yolov9
!python train.py \
--batch 16 --epochs 25 --img 640 --device 0 --min-items 0 --close-mosaic 15 \
--data {dataset.location}/data.yaml \
--weights {HOME}/weights/gelan-c.pt \
--cfg models/detect/gelan-c.yaml \
--hyp hyp.scratch-high.yaml
import torch
import torch.nn as nn
from pathlib import Path

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, H * W)
        value = self.value_conv(x).view(batch_size, -1, H * W)
        attention_map = torch.bmm(query, key)
        attention_map = nn.functional.softmax(attention_map, dim=-1)
        out = torch.bmm(value, attention_map.permute(0, 2, 1)).view(batch_size, C, H, W)
        return self.gamma * out + x
        class YOLOv9WithPostAttention(nn.Module):
    def __init__(self, yolo_model, attention_channels):
        super(YOLOv9WithPostAttention, self).__init__()
        self.yolo_model = yolo_model  # YOLO v9 model instance
        self.post_attention = SelfAttention(in_channels=attention_channels)
        def forward(self, x):
        yolo_output = self.yolo_model(x)  # Run YOLO forward pass
        attention_output = self.post_attention(yolo_output)  # Apply self-attention
        return attention_output
!ls {HOME}/yolov9/runs/train/exp/
from IPython.display import Image
Image(filename=f"{HOME}/yolov9/runs/train/exp5/results.png", width=1000)
%cd {HOME}/yolov9
!python val.py \
--img 640 --batch 32 --conf 0.001 --iou 0.7 --device 0 \
--data {dataset.location}/data.yaml \
--weights {HOME}/yolov9/runs/train/exp5/weights/best.pt
!python detect.py \
--img 1280 --conf 0.1 --device 0 \
--weights {HOME}/yolov9/runs/train/exp5/weights/best.pt \
--source {dataset.location}/test/images
import glob
from IPython.display import Image, display
for image_path in glob.glob(f'{HOME}/yolov9/runs/detect/exp3/*.jpg')[:2]:
      display(Image(filename=image_path, width=700))
    
