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
