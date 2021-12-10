from typing import Tuple, Dict
from torchvision.transforms import transforms, InterpolationMode

from PIL import Image
import numpy as np
import albumentations as A
import imgaug as ia
import imgaug.augmenters as iaa
import cv2


class ToNumpy:

    def __call__(self, x: Image) -> np.ndarray:
        return np.asarray(x)


class FetchImageFromAlbumentationsDict:

    def __call__(self, x: Dict[str, np.ndarray]) -> np.ndarray:
        return x['image']


def train_transforms(
    target_size: Tuple[int, int],
    normalize: bool = True, 
    interpolation_mode = cv2.INTER_NEAREST) -> transforms.Compose:
    def _albument(img):
        return aug_A(image=img)

    aug_A = A.Compose(
                    [
                        A.Resize(target_size[0], target_size[1], interpolation=interpolation_mode),
                        A.Rotate(limit=(-5, 5), p=0.5, interpolation=interpolation_mode),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.GaussianBlur(p=0.3),
                        A.Equalize(by_channels=False, p=0.3)
                    ])
    # aug_ia = iaa.Sometimes(p=1, then_list=iaa.OneOf([
    #             iaa.AdditiveGaussianNoise(),
    #             iaa.LinearContrast(),
    #             iaa.AddToBrightness()
    #          ]))

    transforms_list = [
        ToNumpy(),
        _albument,
        FetchImageFromAlbumentationsDict(),
        # img_aug,
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
