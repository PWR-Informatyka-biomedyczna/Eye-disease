import os
import tqdm
import re
import cv2
import pandas as pd
from PIL import Image


PATH_COLUMN_NAME = 'Path'
# Changed to resized dataset csv
CSV_PATH = '/media/data/adam_chlopowiec/eye_image_classification/pretrain_collected_data_splits.csv'


class Counter:

    def __init__(self):
        self.items = dict()

    def add(self, path: str):
        p = path.split('/')
        data_idx = int(p[[i for i, n in enumerate(p) if n == 'data'][1] + 1])
        if data_idx not in self.items.keys():
            self.items[data_idx] = 1
        else:
            self.items[data_idx] += 1

    def __str__(self) -> str:
        return f'{self.items}'

        


def sanity_check_pil(df: pd.DataFrame):
    counter = Counter()
    for img in tqdm.tqdm(df[PATH_COLUMN_NAME]):
        try:
            ob = Image.open(img)
        except Exception as e:
            counter.add(img)
            print(e)
    return counter


def sanity_check_cv2(df: pd.DataFrame):
    counter = Counter()
    for img in tqdm.tqdm(df[PATH_COLUMN_NAME]):
        ob = cv2.imread(img)
        if ob is None:
            counter.add(img)
            print(img)
    return counter


def sanity_check_pil_pretrain(df: pd.DataFrame):
    counter = Counter()
    for split in ('pretrain', 'preval', 'pretest'):
        df_new = df[df['Split'] == split]
        for img in tqdm.tqdm(df_new[PATH_COLUMN_NAME]):
            try:
                ob = Image.open(img)
            except Exception as e:
                counter.add(img)
                print(e)
    return counter


def main():
    df = pd.read_csv(CSV_PATH)
    pilcounter = sanity_check_pil(df)
    cvcounter = sanity_check_cv2(df)
    #pilcounter = sanity_check_pil_pretrain(df)
    print(f'PIL: {pilcounter}')
    print(f'CV2: {cvcounter}')
    print(f'Total data: {len(df)}')


if __name__ == '__main__':
    main()
