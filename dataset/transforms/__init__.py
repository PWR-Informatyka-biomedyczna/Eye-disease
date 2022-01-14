from typing import Tuple, Dict
from torchvision.transforms import transforms, InterpolationMode

from PIL import Image
import torch
import numpy as np
import albumentations as A
import imgaug as ia
import imgaug.augmenters as iaa
import cv2


class ToNumpy:

    def __call__(self, x: Image) -> np.ndarray:
        return np.asarray(x)

class Albument:

    def __init__(self, augment) -> None:
        self.augment = augment

    def __call__(self, img: Image) -> np.ndarray:
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
                        A.Rotate(limit=(-90, 90), p=0.8, interpolation=interpolation_mode),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        #A.RandomCrop(224, 224, p=1),
                        #A.GaussianBlur(blur_limit=(1,3), p=0.3),
                        #A.GaussianBlur(blur_limit=(1,7), p=0.3),
                        #A.Equalize(by_channels=False, p=0.35),
                        #A.CLAHE(clip_limit=6.0, p=0.35),
                        A.Cutout(num_holes=20, max_h_size=11, max_w_size=11, p=0.5),
                        #A.Affine(translate_percent={'x': (0.03, 0.15), 'y': (0.03, 0.15)}, interpolation=interpolation_mode, p=0.5),
                        #A.GaussNoise(var_limit=(0.0, 5.0), per_channel=False, p=0.5),
                        #A.GaussNoise(var_limit=(0.0, 25.0), per_channel=False, p=0.5),
                        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.3, p=0.5),
                        #A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.15, p=0.5),
                        #A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
                        #A.Sharpen(alpha=(0.05, 0.2), lightness=(0.1, 0.3), p=0.5)
                    ])
    #aug_ia = iaa.Sometimes(1, then_list=[iaa.Sometimes(0.25, iaa.JpegCompression(compression=(25, 50)))])

    albument = Albument(aug_A)
    #imgaugment = Imgaugment(aug_ia)
    transforms_list = [
        transforms.Resize(target_size, interpolation=InterpolationMode.NEAREST),
        ToNumpy(),
        albument,
        #imgaugment,
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
    
    #aug_A = A.Resize(target_size[0], target_size[1], interpolation=interpolation_mode)
    #albument = Albument(aug_A)
    transforms_list = [
        transforms.Resize(target_size, interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor()
    ]
    if normalize:
        transforms_list.append(transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ))
    return transforms.Compose(transforms_list)


