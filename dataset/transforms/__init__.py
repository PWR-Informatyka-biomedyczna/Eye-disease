from typing import Tuple
from torchvision.transforms import transforms, InterpolationMode

import torch
import numpy as np
import albumentations as A
import imgaug as ia
import imgaug.augmenters as iaa
import cv2


class FromNumpy():

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x)


def train_transforms(
    target_size: Tuple[int, int],
    normalize: bool = True, 
    interpolation_mode = cv2.INTER_NEAREST) -> transforms.Compose:
    aug_A = A.Compose(
                    [
                        A.Resize(target_size[0], target_size[1], interpolation=interpolation_mode),
                        A.Rotate(limit=(-5, 5), p=0.5, interpolation=interpolation_mode),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.GaussianBlur(p=0.3),
                        A.Equalize(by_channels=False, p=0.3)
                    ])
    aug_ia = iaa.Sometimes(p=1, then_list=iaa.OneOf([
                iaa.AdditiveGaussianNoise(),
                iaa.LinearContrast(),
                iaa.AddToBrightness()
             ]))

    transforms_list = [
        aug_A,
        aug_ia,
        transforms.ToTensor()
    ]
    if normalize:
        transforms_list.append(transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ))
    return transforms.Compose(transforms_list)

def test_val_transforms(
    target_size: Tuple[int, int], 
    normalize: bool = True,
    interpolation_mode: InterpolationMode = InterpolationMode.NEAREST) -> transforms.Compose:

    transforms_list = [
        transforms.Resize(target_size, interpolation=interpolation_mode),
        transforms.ToTensor()
    ]
    if normalize:
        transforms_list.append(transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ))
    return transforms.Compose(transforms_list)
