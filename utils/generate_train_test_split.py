from typing import Tuple

import pandas as pd 
from sklearn.model_selection import train_test_split


CSV_PATH = '/media/data/adam_chlopowiec/eye_image_classification/collected_data.csv'
NEW_CSV_PATH = '/media/data/adam_chlopowiec/eye_image_classification/collected_data_splits.csv'
SPLIT_RATIOS = {
    'train_dev': 0.85,
    'train': 0.85
}
LABEL_COLUMN_NAME = 'Label'
SPLIT_COLUMN_NAME = 'Split'

TRAIN_SPLIT_NAME = 'train'
DEV_SPLIT_NAME = 'val'
TEST_SPLIT_NAME = 'test'




def split_data(df: pd.DataFrame, train_ratio: float, stratify: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train, df_test = train_test_split(df, train_size = train_ratio, stratify=df[stratify])
    return df_train, df_test


def create_split():
    df_old = pd.read_csv(CSV_PATH)
    df_train_val, df_test = split_data(
        df_old, 
        SPLIT_RATIOS['train_dev'], 
        stratify=LABEL_COLUMN_NAME
    )
    df_train, df_val = split_data(
        df_train_val, 
        SPLIT_RATIOS['train'], 
        stratify=LABEL_COLUMN_NAME
    )
    df_train[SPLIT_COLUMN_NAME] = TRAIN_SPLIT_NAME
    df_val[SPLIT_COLUMN_NAME] = DEV_SPLIT_NAME
    df_test[SPLIT_COLUMN_NAME] = TEST_SPLIT_NAME
    df_new = pd.concat([df_train, df_val, df_test])
    df_new.to_csv(NEW_CSV_PATH)
    print(f"""
Splits created:
======================================================
Original dataset size: {len(df_old)}
Splits ratios: {df_new.groupby(SPLIT_COLUMN_NAME).count()}
=======================================================
New dataset size: {len(df_new)}
    """)
    


if __name__ == '__main__':
    create_split()
