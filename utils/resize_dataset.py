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
        new_path = path.replace("eye_image_classification/data", "eye_image_classification/data_resized")
        new_paths.append(new_path)
    return new_paths

def transform_copy_img(path, new_path):
    img = Image.open(path)
    img = np.asarray(img)
    img = A.Resize(380, 380, interpolation=cv2.INTER_NEAREST)(image=img)['image']
    img = Image.fromarray(img)
    img.save(new_path)

data = pd.read_csv('/media/data/adam_chlopowiec/eye_image_classification/collected_data.csv')
paths = data['Path']
#print(data.head(5))

new_paths = change_dirs(paths)
#for path in paths:
#    print(path)
#make_dirs(new_paths)
#for path, new_path in tqdm.tqdm(zip(paths, new_paths)):
#    transform_copy_img(path, new_path)

data['Path'] = new_paths
data.to_csv('../resized_collected_data.csv')