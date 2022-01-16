import albumentations as A
import pandas as pd
import tqdm
import cv2
import os
import numpy as np
from PIL import Image
from pathlib import Path


def make_dirs(paths):
    for path in paths:
        dir_path = os.path.dirname(path)
        Path(dir_path).mkdir(mode=777, parents=True, exist_ok=True)
    return

def change_dirs(paths):
    new_paths = []
    for path in paths:
        new_path = path.replace("/home/adam_chlopowiec/data/eye_image_classification/data_resized", "../input/datasetresized/data_resized")
        new_paths.append(new_path)
    return new_paths

def transform_copy_img(path, new_path):
    img = Image.open(path)
    img = np.asarray(img)
    img = A.Resize(224, 224, interpolation=cv2.INTER_NEAREST)(image=img)['image']
    img = Image.fromarray(img)
    img.save(new_path)

data = pd.read_csv('pretrain_collected_data_splits.csv')
paths = data['Path']

new_paths = change_dirs(paths)
#make_dirs(new_paths)
#for path, new_path in tqdm.tqdm(zip(paths, new_paths)):
#    transform_copy_img(path, new_path)

data['Path'] = new_paths
data.to_csv('kaggle_collected_data_splits.csv')