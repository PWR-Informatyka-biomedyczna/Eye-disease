from typing import Callable
import pandas as pd


def identity_resampler(df: pd.DataFrame) -> pd.DataFrame:
    return df


def threshold_to_dr_with_ros(df):
    df_train = df[df['Split'] == 'train']
    new_df = pd.DataFrame(columns=df_train.columns)
    classes = df_train.value_counts('Label', sort=True)
    dr_class = classes[3]
    for cls, _ in classes.items():
        cls_df = df_train[df_train['Label'] == cls]
        class_count = dr_class
        if cls == 0:
            resampled = cls_df.sample(class_count, replace=False, ignore_index=True)
        elif cls == 3:
            resampled = cls_df
        else:
            resampled = cls_df.sample(class_count, replace=True, ignore_index=True)
        new_df = pd.concat([new_df, resampled], ignore_index=True)
    val_df = df[df['Split'] == 'val']
    test_df = df[df['Split'] == 'test']
    new_df = pd.concat([new_df, val_df], ignore_index=True)
    new_df = pd.concat([new_df, test_df], ignore_index=True)
    return new_df


def threshold_to_glaucoma_with_ros(df):
    df_train = df[df['Split'] == 'train']
    new_df = pd.DataFrame(columns=df_train.columns)
    classes = df_train.value_counts('Label', sort=True)
    glacoma_class = classes[1]
    for cls, _ in classes.items():
        cls_df = df_train[df_train['Label'] == cls]
        class_count = glacoma_class
        if cls == 0 or cls == 3:
            resampled = cls_df.sample(class_count, replace=False, ignore_index=True)
        elif cls == 1:
            resampled = cls_df
        else:
            resampled = cls_df.sample(class_count, replace=True, ignore_index=True)
        new_df = pd.concat([new_df, resampled], ignore_index=True)
    val_df = df[df['Split'] == 'val']
    test_df = df[df['Split'] == 'test']
    new_df = pd.concat([new_df, val_df], ignore_index=True)
    new_df = pd.concat([new_df, test_df], ignore_index=True)
    return new_df


def binary_thresh_to_20k_equal(df):
    COUNT = 10000
    df_train = df[df['Split'] == 'train']
    new_df = pd.DataFrame(columns=df_train.columns)
    classes = df_train.value_counts('Label', sort=True)
    class_1_count = classes[1]
    class_2_count = classes[2]
    class_3_count = COUNT - class_1_count - class_2_count
    for cls, _ in classes.items():
        class_df = df_train[df_train['Label'] == cls]
        if cls == 0:
            class_df = class_df.sample(COUNT, replace=False, ignore_index=True)
        elif cls == 3:
            class_df = class_df.sample(class_3_count, replace=False, ignore_index=True)
        new_df = pd.concat([new_df, class_df], ignore_index=True)

    val_df = df[df['Split'] == 'val']
    test_df = df[df['Split'] == 'test']
    new_df = pd.concat([new_df, val_df], ignore_index=True)
    new_df = pd.concat([new_df, test_df], ignore_index=True)
    
    return new_df