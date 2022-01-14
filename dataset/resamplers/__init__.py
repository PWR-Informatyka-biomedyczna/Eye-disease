from typing import Callable
import pandas as pd


def identity_resampler(df: pd.DataFrame) -> pd.DataFrame:
    return df


def to_lowest_resampler(target_label: str = 'Label', train_split_name: str = 'train') -> Callable:
    def _resampler(df: pd.DataFrame) -> pd.DataFrame:
        df_train = df[df['Split'] == train_split_name]
        classes = df_train.value_counts(target_label, sort=True)
        max_classes = classes.max()
        for cls, num_samples in classes.items():
            cls_df = df_train[df_train[target_label] == cls]
            difference = max_classes - num_samples
            resampled = cls_df.sample(difference, replace=True, ignore_index=True)
            df = pd.concat([df, resampled], ignore_index=True)
        return df
    return _resampler


def threshold_to_dr_with_ros(df):
    df_train = df[df['Split'] == 'train']
    new_df = pd.DataFrame(columns=df_train.columns)
    classes = df_train.value_counts('Label', sort=True)
    dr_class = classes[3]
    for cls, num_samples in classes.items():
        cls_df = df_train[df_train['Label'] == cls]
        class_count = dr_class
        if cls == 0 or cls == 3:
            resampled = cls_df.sample(class_count, replace=False, ignore_index=True)
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
    for cls, num_samples in classes.items():
        cls_df = df_train[df_train['Label'] == cls]
        class_count = glacoma_class
        if cls == 0 or cls == 1 or cls == 3:
            resampled = cls_df.sample(class_count, replace=False, ignore_index=True)
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
    class_0_df = df_train[df_train['Label'] == 0]
    class_1_df = df_train[df_train['Label'] == 1]
    class_2_df = df_train[df_train['Label'] == 2]
    class_3_df = df_train[df_train['Label'] == 3]
    class_3_df_resampled = class_3_df.sample(class_3_count, replace=False, ignore_index=True)
    class_0_df_resampled = class_0_df.sample(COUNT, replace=False, ignore_index=True)
    new_df = pd.concat([new_df, class_0_df_resampled], ignore_index=True)
    new_df = pd.concat([new_df, class_1_df], ignore_index=True)
    new_df = pd.concat([new_df, class_2_df], ignore_index=True)
    new_df = pd.concat([new_df, class_3_df_resampled], ignore_index=True)
    val_df = df[df['Split'] == 'val']
    test_df = df[df['Split'] == 'test']
    new_df = pd.concat([new_df, val_df], ignore_index=True)
    new_df = pd.concat([new_df, test_df], ignore_index=True)
    
    return new_df