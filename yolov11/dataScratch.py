import torch
from PIL import Image
import open_clip
available_models = open_clip.list_pretrained()
print(available_models)
from datasets import load_dataset

import os
# 设置镜像站点
os.environ['HF_HOME'] = "/data1/liuzelin/fish"  # 替换成你想要的路径
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
cache_dir = "/data1/liuzelin/fish"  # 替换成你想要的路径

# 加载数据集时指定缓存目录
ds = load_dataset(
    "imageomics/fish-vista", 
    "species_classification",
    cache_dir=cache_dir
)