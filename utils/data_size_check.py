import albumentations as A
import pandas as pd
import tqdm
import cv2
import os
import numpy as np
from PIL import Image
from pathlib import Path


SIZE = (380, 380)
CSV_PATH = '/media/data/adam_chlopowiec/eye_image_classification/resized_collected_data.csv'

def load_and_check_image(path):
    img = Image.open(path)
    width, height = img.size
    if width != SIZE[0] and height != SIZE[1]:
        print('DUPA')

data = pd.read_csv(CSV_PATH)
paths = data['Path']

for path in tqdm.tqdm(paths):
    load_and_check_image(path)
