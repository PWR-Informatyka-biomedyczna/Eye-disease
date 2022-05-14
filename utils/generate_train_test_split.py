from typing import Tuple

import pandas as pd 
from sklearn.model_selection import train_test_split


CSV_PATH = r'C:\Users\Adam\Desktop\Studia\Psy Tabakowa\eye-disease\Eye-disease\pretrain_corrected_data_splits.csv'
NEW_CSV_PATH = r'C:\Users\Adam\Desktop\Studia\Psy Tabakowa\eye-disease\Eye-disease\pretrain_no_eyepacs_corrected_data_splits.csv'
PRETRAIN_CSV_PATH = r'C:\Users\Adam\Desktop\Studia\Psy Tabakowa\eye-disease\Eye-disease\pretrain_no_eyepacs_corrected_data_splits.csv'
SPLIT_RATIOS = {
    'train_dev': 0.80,
    'train': 0.80,
}
LABEL_COLUMN_NAME = 'Label'
SPLIT_COLUMN_NAME = 'Split'

TRAIN_SPLIT_NAME = 'train'
DEV_SPLIT_NAME = 'val'
TEST_SPLIT_NAME = 'test'
PRETRAIN_SPLIT_NAME = 'pretrain'

def split_data(df: pd.DataFrame, train_ratio: float, stratify: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train, df_test = train_test_split(df, train_size=train_ratio, stratify=df[stratify])
    return df_train, df_test


def create_split(df_old: pd.DataFrame):
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
    

def create_train_val_test_split(df_old: pd.DataFrame):
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
    return df_new


def get_class_counts(df: pd.DataFrame, ordered_labels):
    labels = df[LABEL_COLUMN_NAME]
    counts = labels.value_counts()
    print(counts)
    class_counts = {}
    for class_, count in zip(ordered_labels, counts):
        class_counts[class_] = count
    return class_counts


def equalize_class_counts(class_counts, min_count):
    min_class_count = list(class_counts.values())[-1]
    if min_count < min_class_count:
        min_count = min_class_count
    pretrain_counts = {}
    for key, value in class_counts.items():
        pretrain_counts[key] = value - min_count
    return pretrain_counts


def get_pretrain_class_counts(pretrain_counts, classes):
    pretrain_class_counts = {}
    for class_ in classes:
        pretrain_class_counts[class_] = pretrain_counts[class_]
    return pretrain_class_counts


def get_pretrain_ratios(class_counts, pretrain_counts):
    class_ratios = {}
    for class_ in pretrain_counts.keys():
        class_count = class_counts[class_]
        pretrain_class_count = pretrain_counts[class_]
        ratio = pretrain_class_count / class_count
        class_ratios[class_] = ratio
    return class_ratios


def remove_eyepacs(df):
    df = df[df['Dataset'] != 9]
    return df


def train_val_test_pretrain_preval_pretest_split(classes=[0, 1, 2 ,3], pretrain_classes=[0, 3]):
    df_old = pd.read_csv(CSV_PATH)
    pre_train_dfs = {}
    train_dfs = {}
    class_counts = get_class_counts(df_old, [0, 3, 1, 2])
    pretrain_counts = equalize_class_counts(class_counts, class_counts[1])
    pretrain_counts = get_pretrain_class_counts(pretrain_counts, pretrain_classes)
    class_ratios = get_pretrain_ratios(class_counts, pretrain_counts)
    for class_ in pretrain_classes:
        pre_train_dfs[class_], train_dfs[class_] = train_test_split(df_old[df_old[LABEL_COLUMN_NAME] == class_], train_size=class_ratios[class_])
    
    train_val_test_classes = [class_ for class_ in classes if class_ not in pretrain_classes]
    for class_ in train_val_test_classes:
        train_dfs[class_] = df_old[df_old[LABEL_COLUMN_NAME] == class_]
    df_to_train_test_split = pd.concat(train_dfs.values())
    df_split = create_train_val_test_split(df_to_train_test_split)
    df_pretrain_concat = pd.concat(pre_train_dfs.values())

    df_pretrain_val, df_pretest = split_data(
        df_pretrain_concat, 
        SPLIT_RATIOS['train_dev'], 
        stratify=LABEL_COLUMN_NAME
    )

    df_pretrain, df_preval = split_data(
        df_pretrain_val, 
        SPLIT_RATIOS['train'], 
        stratify=LABEL_COLUMN_NAME
    )

    df_pretrain[SPLIT_COLUMN_NAME] = PRETRAIN_SPLIT_NAME
    # df_preval[SPLIT_COLUMN_NAME] = PRETRAIN_DEV_SPLIT_NAME
    # df_pretest[SPLIT_COLUMN_NAME] = PRETRAIN_TEST_SPLIT_NAME
    df_new_pretrain = pd.concat([df_pretrain, df_preval, df_pretest])
    dataset = pd.concat([df_split, df_new_pretrain])
    dataset.to_csv(PRETRAIN_CSV_PATH)
    print(f"""
Splits created:
======================================================
Original dataset size: {len(df_old)}
Splits ratios: {dataset.groupby(SPLIT_COLUMN_NAME).count()}
=======================================================
New dataset size: {len(dataset)}
    """)


def train_val_test_pretrain_split(classes=[0, 1, 2 ,3], pretrain_classes=[0, 3]):
    df_very_old = pd.read_csv(CSV_PATH)
    df_old = pd.read_csv(CSV_PATH)
    df_old = df_old.drop(columns=[SPLIT_COLUMN_NAME])
    df_old = remove_eyepacs(df_old)
    pre_train_dfs = {}
    train_dfs = {}
    class_counts = get_class_counts(df_old, [0, 3, 1, 2])
    pretrain_counts = equalize_class_counts(class_counts, class_counts[1])
    pretrain_counts = get_pretrain_class_counts(pretrain_counts, pretrain_classes)
    class_ratios = get_pretrain_ratios(class_counts, pretrain_counts)
    for class_ in pretrain_classes:
        pre_train_dfs[class_], train_dfs[class_] = train_test_split(df_old[df_old[LABEL_COLUMN_NAME] == class_], train_size=class_ratios[class_])
    
    train_val_test_classes = [class_ for class_ in classes if class_ not in pretrain_classes]
    for class_ in train_val_test_classes:
        train_dfs[class_] = df_old[df_old[LABEL_COLUMN_NAME] == class_]
    df_to_train_test_split = pd.concat(train_dfs.values())
    df_train, df_val = split_data(
        df_to_train_test_split,
        SPLIT_RATIOS['train_dev'],
        stratify=LABEL_COLUMN_NAME
    )
    df_train[SPLIT_COLUMN_NAME] = TRAIN_SPLIT_NAME
    df_val[SPLIT_COLUMN_NAME] = DEV_SPLIT_NAME
    test_df[SPLIT_COLUMN_NAME] = TEST_SPLIT_NAME
    #df_split = create_train_val_test_split(df_to_train_test_split)
    
    df_pretrain = pd.concat(pre_train_dfs.values())
    df_pretrain[SPLIT_COLUMN_NAME] = PRETRAIN_SPLIT_NAME
    dataset = pd.concat([df_train, df_val, test_df, df_pretrain])
    dataset.to_csv(PRETRAIN_CSV_PATH)
    print(f"""
Splits created:
======================================================
Original dataset size: {len(df_very_old)}
Splits ratios: {dataset.groupby([SPLIT_COLUMN_NAME, 'Label']).count()}
=======================================================
New dataset size: {len(dataset)}
    """)


if __name__ == '__main__':
    #create_split()
    train_val_test_pretrain_split()
