import pandas as pd


def change_dirs(paths):
    new_paths = []
    for path in paths:
        new_path = path.replace("/home/adam_chlopowiec/data/eye_image_classification/full_data_resized/", "../input/datacorrected/full_data_resized/")
        new_paths.append(new_path)
    return new_paths

data = pd.read_csv('pretrain_no_eyepacs_corrected_data_splits.csv')
paths = data['Path']

new_paths = change_dirs(paths)
data['Path'] = new_paths
data.to_csv('kaggle_no_eyepacs_pretrain_corrected_data_splits.csv')