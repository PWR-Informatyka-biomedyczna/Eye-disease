import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image
import numpy as np

img = Image.open(r'D:\KNBMI\data\5\DR2-images-by-lesions\Normal Images\IM000102.JPG')
aug_ia = iaa.Sometimes(p=1, then_list= [
                                    iaa.Sometimes(p=0.2, then_list=[iaa.AdditiveGaussianNoise()]),
                                    iaa.Sometimes(p=0.3, then_list=[iaa.LinearContrast()]),
                                    iaa.Sometimes(p=0.3, then_list=[iaa.AddToBrightness()])])
img = np.asarray(img)
img = aug_ia(images=[img])
print(img[0])