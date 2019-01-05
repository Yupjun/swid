#!/usr/bin/env python
# coding: utf-8

# In[1]:


import imgaug
import cv2

"""
사용 방법:
generator에
바꾸고 싶은 이미지가 들어가 있는 폴더를 지정한다.

"""

def generator(image_list):

    for name in image_list:

        images = cv2.imread(name)
        
        
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        seq = iaa.Sequential([
            ##flip vertical
            iaa.Flipud(0.5),


            iaa.Fliplr(0.5), # horizontal flips
            iaa.Crop(percent=(0, 0.1)), # random crops
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.

            iaa.Multiply((0.9, 1.1)),
            iaa.Sometimes(0.5,
                iaa.GaussianBlur(sigma=(0, 1))
            ),
            # Strengthen or weaken the contrast in each image.
            iaa.ContrastNormalization((0.75, 1.5)),

            sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),

            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
            iaa.Multiply((0.8, 1.2), per_channel=0.2),

            iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25),

            iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-25, 25),
                shear=(-8, 8)
            ),

            iaa.Dropout((0.01, 0.1), per_channel=0.5),
        ], random_order=False) # apply augmenters in random order

        ## range에 따라 이미지 생성
        ## 이미지를 생성
        for i in range(10):

            images_aug = seq.augment_image(images)
            name = 'wafer' + str(i)+'.jpg'
            cv2.imwrite(name, images_aug)

        
generator(["dummy.jpg"])


# In[104]:


type(images_aug)
images_aug.shape


# In[107]:


cv2.imwrite('color_img.jpg', images_aug)
cv2.imshow("image", images_aug);
cv2.waitKey();


# In[ ]:




