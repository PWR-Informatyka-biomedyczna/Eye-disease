import pandas as pd

CSV_PATH_1 = '/media/data/adam_chlopowiec/eye_image_classification/collected_data.csv'
CSV_PATH_2 = '/media/data/adam_chlopowiec/eye_image_classification/resized_collected_data.csv'

data_original = pd.read_csv(CSV_PATH_1)
data_copied = pd.read_csv(CSV_PATH_2)

copied_paths = data_copied['Path']
changed_copied_paths = []
for path in copied_paths:
    path = path.replace('eye_image_classification/data_resized', 'eye_image_classification/data')
    changed_copied_paths.append(path)

original_paths = data_original['Path']
original_labels = data_original['Label']
copied_labels = data_copied['Label']
for original_path, copied_path in zip(original_paths, changed_copied_paths):
    if original_path != copied_path:
        print(f'Original Path: {original_path}')
        print(f'Copied Path: {copied_path}')

for original_label, copied_label in zip(original_labels, copied_labels):
    if original_label != copied_label:
        print(f'Original Label: {original_label}')
        print(f'Copied Label: {copied_label}')
