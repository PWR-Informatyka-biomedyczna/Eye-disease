import pandas as pd
import os

def change_dirs(paths):
    new_paths = []
    for path in paths:
        new_path = path.replace("/home/adam_chlopowiec/data/eye_image_classification/full_data_resized/",
                                "C:/Users/Adam/Desktop/Studia/Psy Tabakowa/eye-disease/data/full_data_resized/")
        new_path = os.path.abspath(new_path)
        new_paths.append(new_path)
    return new_paths

data = pd.read_csv(r'C:\Users\Adam\Desktop\Studia\Psy Tabakowa\eye-disease\data\pretrain_corrected_data_splits.csv')
paths = data['Path']

new_paths = change_dirs(paths)
data['Path'] = new_paths
data.to_csv(r'C:\Users\Adam\Desktop\Studia\Psy Tabakowa\eye-disease\data\pretrain_corrected_data_splits_windows.csv')
# df = pd.read_csv(r'C:\Users\Adam\Desktop\Studia\Psy Tabakowa\eye-disease\data\pretrain_corrected_data_splits.csv')
# print(df.groupby(["Split", "Label"]).count())