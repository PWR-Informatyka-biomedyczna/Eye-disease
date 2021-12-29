import pandas as pd


CSV_PATH = '/media/data/adam_chlopowiec/eye_image_classification/pretrain_collected_data_splits.csv'


def test():
    df = pd.read_csv(CSV_PATH)
    #print(df['Label'].value_counts())
    splits = df['Split'].unique()
    for split in splits:
        classes_in_split = df[df['Split'] == split]['Label'].value_counts()
        print(split)
        print(classes_in_split)


if __name__ == "__main__":
    test()