from typing import Tuple, Dict
from torchvision.transforms import transforms

from PIL import Image
import numpy as np
import albumentations as A
import cv2
from timm.data import rand_augment_transform


class ToNumpy:

    def __call__(self, x: Image) -> np.ndarray:
        return np.asarray(x)

class Albument:

    def __init__(self, augment) -> None:
        self.augment = augment

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.augment(image=img)['image']

class Imgaugment:

    def __init__(self, augment) -> None:
        self.augment = augment

    def __call__(self, img: np.ndarray) -> np.ndarray:
        augmented_images = self.augment(images=[img])
        return augmented_images[0]


def train_transforms(
    target_size: Tuple[int, int],
    normalize: bool = True, 
    interpolation_mode = cv2.INTER_NEAREST) -> transforms.Compose:

    aug_A = A.Compose(
                    [
                        A.Resize(target_size[0], target_size[1], interpolation=interpolation_mode),
                        A.Rotate(limit=(-90, 90), p=0.8, interpolation=interpolation_mode),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.15, p=0.5),
                        #A.CLAHE(clip_limit=3.0, p=0.5),
                        A.Cutout(num_holes=20, max_h_size=11, max_w_size=11, p=0.5),
                    ])

    albument = Albument(aug_A)

    transforms_list = [
        ToNumpy(),
        albument,
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
    interpolation_mode = cv2.INTER_NEAREST) -> transforms.Compose:
    
    albument = Albument(
        A.Resize(target_size[0], target_size[1], interpolation=interpolation_mode)
    )
    transforms_list = [
        ToNumpy(),
        albument,
        transforms.ToTensor()
    ]
    if normalize:
        transforms_list.append(transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ))
    return transforms.Compose(transforms_list)

def mpl_transforms(
    target_size: Tuple[int, int], 
    normalize: bool = True,
    interpolation_mode = cv2.INTER_NEAREST) -> transforms.Compose:
    
    aug_A = A.Compose(
        [
            A.Resize(target_size[0], target_size[1], interpolation=interpolation_mode), 
            A.HorizontalFlip(p=0.5), 
            A.VerticalFlip(p=0.5)
        ]
    )
    
    albument = Albument(
        aug_A
    )
    config_str = "rand-m6-n2-mstd0.5"
    hparams = {}
    rand_augment = rand_augment_transform(config_str, hparams)
    transforms_list = [
        ToNumpy(),
        albument,
        rand_augment,
        transforms.ToTensor()
    ]
    if normalize:
        transforms_list.append(transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ))
    return transforms.Compose(transforms_list)

