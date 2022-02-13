import pandas as pd


def change_dirs(paths):
    new_paths = []
    for path in paths:
        new_path = path.replace("../input/datasetresized/data_resized/data_resized", "../input/datasetresized/data_resized")
        new_paths.append(new_path)
    return new_paths

data = pd.read_csv('kaggle_pretrain_collected_data_splits_base.csv')
paths = data['Path']

new_paths = change_dirs(paths)
data['Path'] = new_paths
data.to_csv('kaggle_pretrain_collected_data_splits2.csv')