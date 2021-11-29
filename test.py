import albumentations as A
import imgaug as ia
import cv2
import numpy as np

img = cv2.imread(r'D:\semestr_5_adam\dotnet\lista3\img\pikachu.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
"""
aug = A.Compose(
    [
        A.Resize(512, 512),
        #A.Rotate(limit=5, p=1, interpolation=cv2.INTER_NEAREST)
        #A.SafeRotate(limit=5, p=1, interpolation=cv2.INTER_NEAREST)
        #A.HorizontalFlip(p=1),
        #A.VerticalFlip(p=1),
        #A.GaussianBlur(blur_limit=(5, 5), p=1),
        #A.GaussNoise(var_limit=(0, 100), mean=0, per_channel=True, p=1),
        #A.Equalize(by_channels=False, p=1),
        #A.CLAHE(clip_limit=(5, 5), tile_grid_size=(40, 40), p=1),
        #A.ElasticTransform(p=1, interpolation=cv2.INTER_NEAREST)
    ]
)

cv2.namedWindow('win', cv2.WINDOW_AUTOSIZE)

for _ in range(10):
    new_img = aug(image=img)['image']
    #print(img['image'])
    cv2.imshow('win', new_img)
    cv2.waitKey(0)
"""
#aug = ia.augmenters.AverageBlur(k=(3,5))
aug = ia.augmenters.AdditiveGaussianNoise()
#aug = ia.augmenters.Add()
aug = ia.augmentersa.LinearContrast()
#aug = ia.augmenters.CLAHE()
#aug = ia.augmenters.AllChannelsCLAHE()
#aug = ia.augmenters.HistogramEqualization()
#aug = ia.augmenters.AllChannelsHistogramEqualization()
#aug = ia.augmenters.PerspectiveTransform()
aug = ia.augmenters.AddToBrightness()
#aug = ia.augmenters.MultiplyBrightness()
#aug = ia.augmenters.Multiply(mul=(0.7, 1.3))
#aug = ia.augmenters.KMeansColorQuantization()


img = A.Resize(512, 512)(image=img)['image']

for _ in range(10):
    new_img = aug(images=[img])
    cv2.imshow('win', new_img[0])
    cv2.waitKey(0)

aug = A.Compose(
    [
        A.Resize(512, 512),
        A.Rotate(limit=5, p=1, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=1),
        A.VerticalFlip(p=1),
        A.GaussianBlur(blur_limit=(5, 5), p=1),
        A.Equalize(by_channels=False, p=1)
    ]
)


aug = ia.augmenters.AdditiveGaussianNoise()
aug = ia.augmentersa.LinearContrast()
aug = ia.augmenters.AddToBrightness()
