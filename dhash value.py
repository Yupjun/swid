#!/usr/bin/env python
# coding: utf-8

# In[1]:


#분포 확인할 것

import random
import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('crack_all/1.jpg',0)
img2 = cv2.imread('no_crack_all/4346.jpg',0)

img1 = cv2.resize(img1,(1280,30))
img2 = cv2.resize(img2,(1280,30))



plt.hist(img1.flatten(),256,[0,256], color = 'r')
plt.hist(img2.flatten(),256,[0,256], color = 'b')
plt.xlim([0,50])
plt.legend(('crack_image','no_crack_image'), loc = 'upper right')
plt.show()


# In[2]:


import dhash
from PIL import Image

image = Image.open('crack_all/1.jpg')
row, col = dhash.dhash_row_col(image)
print(dhash.format_hex(row, col))

image = Image.open('crack_all/2.jpg')
row, col = dhash.dhash_row_col(image)
print(dhash.format_hex(row, col))


# In[2]:


import random
import cv2
import numpy as np

import dhash
from PIL import Image

"""
데이터 수집해서
CRACK_VALUE와 NO_CRACK_VALUE의 값을 채워야 한다.
현재는 dummy variable
"""

CRACK_VALUE= 1
NO_CRACK_VALUE = 0

def dhash_value(name):

    img1 = cv2.imread(name,0)
    img1 = cv2.resize(img1,(9,9))

    tmp_list = img1.tolist()

    temp=[]
    temp_value=0
    t1=""
    for i in tmp_list:
        temps =[]
        for k in range(len(i)-1):
            if(i[k+1]>i[k]):
                temps.append(1)
                temp_value+=1
                t1+=str(1)
            else:
                temps.append(0)
                t1+=str(0)
                
        temp.append(temps)



    temp1 =[]
   
    ans =0
#     for k in range(len(t1)):
#         if int(t1[k]) != int(t2[k]):
#             ans+=1

    print("crack1 hash value : " + t1)
    print("hamming distance : %d" %(temp_value))
    
dhash_value("dummy.jpg")

