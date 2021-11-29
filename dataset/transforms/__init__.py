from typing import Tuple
from torchvision.transforms import transforms, InterpolationMode

import albumentations as A
import imgaug as ia
import imgaug.augmenters as iaa
import cv2


def train_transforms(
    target_size: Tuple[int, int],
    normalize: bool = True, 
    interpolation_mode: InterpolationMode = InterpolationMode.NEAREST) -> transforms.Compose:
    transforms_list = [
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
        transforms.ToTensor()
    ]
    if normalize:
        transforms_list.append(transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ))
    return transforms.Compose(transforms_list)

def extended_train_transforms(
        target_size: Tuple[int, int], 
        interpolation_mode = cv2.INTER_NEAREST) -> transforms.Compose:
    
    aug_A = A.Compose(
                    [
                        A.Resize(target_size[0], target_size[1]),
                        A.Rotate(limit=(-5, 5), p=0.5, interpolation=interpolation_mode),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.GaussianBlur(blur_limit=(5, 5), p=0.3),
                        A.Equalize(by_channels=False, p=0.2)
                    ])
    aug_ia = iaa.OneOf([
                iaa.AdditiveGaussianNoise(),
                iaa.LinearContrast(),
                iaa.AddToBrightness()
             ])
    # Tune parametry tych augmentacji
    return aug_A, aug_ia
