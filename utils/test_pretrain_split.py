import pandas as pd


CSV_PATH = r'C:\Users\Adam\Desktop\Studia\Psy Tabakowa\eye-disease\Eye-disease\pretrain_collected_data.csv'


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