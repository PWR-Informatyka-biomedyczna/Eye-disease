import pandas as pd
import os
from pathlib import Path


def make_dirs(paths):
    for path in paths:
        dir_path = os.path.dirname(path)
        Path(dir_path).mkdir(mode=777, parents=True, exist_ok=True)
    return

def change_dirs(paths):
    new_paths = []
    for path in paths:
        new_path = path.replace("/home/adam_chlopowiec/data/eye_image_classification/data_resized", "../input/data-resized/data_resized")
        new_paths.append(new_path)
    return new_paths


data = pd.read_csv('pretrain_collected_data_splits.csv')
paths = data['Path']

new_paths = change_dirs(paths)

data['Path'] = new_paths
data.to_csv('kaggle_pretrain_collected_data_splits.csv')