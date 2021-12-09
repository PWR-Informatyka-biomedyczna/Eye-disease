import torch
import cv2

from methods import ResNet18Model
from dataset import EyeDiseaseDataModule, resamplers
from dataset.transforms import test_val_transforms, train_transforms


CHECKPOINT = '/home/adam_chlopowiec/data/eye_image_classification/Eye-disease/checkpoints/epoch=6-step=7356.ckpt'


def main():
    model = ResNet18Model.load_from_checkpoint(CHECKPOINT)
    data_module = EyeDiseaseDataModule(
    csv_path='/media/data/adam_chlopowiec/eye_image_classification/collected_data_splits.csv',
    train_split_name='train',
    val_split_name='val',
    test_split_name='test',
    train_transforms=train_transforms((224, 224), True, cv2.INTER_NEAREST),
    val_transforms=test_val_transforms((224, 224), True, cv2.INTER_NEAREST),
    test_transforms=test_val_transforms((224, 224), True, cv2.INTER_NEAREST),
    image_path_name='Path',
    target_name='Label',
    split_name='Split',
    batch_size=16,
    num_workers=1,
    shuffle_train=True,
    resampler=resamplers.to_lowest_resampler(
        target_label='Label',
        train_split_name='train')
    )
    data_module.prepare_data()

    print('dupa')
    preds = {}
    with torch.no_grad():
        for batch in data_module.test_dataloader():
            print(batch)


if __name__ == '__main__':
    main()
