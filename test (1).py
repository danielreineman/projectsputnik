from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab
import random
import json
import os
from PIL import Image
import numpy.ma as npm
from skimage import measure,draw
import cv2
#import .pixel_lat_long

def delete_zero_bfstr(ss):
    for i in range(len(ss)):
        if ss[i]=='0':
            continue
        else:
            ss=ss[i:]
            break
    return ss

def find_id_ann(ann,imgid):
    l=[]
    for anni in ann:
        if str(anni['image_id'])==imgid:
            l.append(anni)
    return l

def plotall():
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.subplot(1,3,2)
    plt.imshow(img_real)
    plt.subplot(1,3,3)
    plt.imshow(full_img)

with open('/home/derose/Downloads/boston.json','r') as f:
    prediction_json=json.load(f)

#testimages_dir='/home/derose/osmc/data/_test_images'
testimages_dir='/home/derose/Downloads/Boston'
testimages_list=os.listdir(testimages_dir)

    #image_id=random.choice(testimages_list)
mask=np.zeros([3300,3300])
full_img=np.uint8(np.zeros([3300,3300,3]))
for image_id in testimages_list:#[0:100]:
    #image_id=random.choice(testimages_list)
    img_filepath=os.path.join(testimages_dir,image_id)
    img=mpimg.imread(img_filepath)
    img_real=mpimg.imread(img_filepath)    
    img_id=delete_zero_bfstr(image_id.split('.')[0])
    x=int(float(img_id.split('_')[0])-LLcorner[0])
    y=975-int(float(img_id.split('_')[1])-LLcorner[1])
    img_annlist=find_id_ann(prediction_json,img_id)    
    #full_img[(y-1)*300:y*300,(x-1)*300:x*300,0:3]=img_real
    full_img[y*3:y*3+300,x*3:x*3+300,0:3]=img_real
    for ann in img_annlist:
        m=cocomask.decode(ann['segmentation'])        
        #mask[(y-1)*300:y*300,(x-1)*300:x*300]+=m
        mask[y*3:y*3+300,x*3:x*3+300]+=m
        #mask+=m
#mask=np.floor(mask/4)
img=full_img
img_real=full_img
mask=mask>0
contours = measure.find_contours(mask, 0.5)
img.flags.writeable=True

plt.figure()
plt.subplot(1,2,1)
plt.title('original image')
plt.imshow(img_real)
plt.axis('off')
plt.subplot(1,2,2)
plt.title('masked image')
plt.imshow(img)
for n, contour in enumerate(contours):
    plt.plot(contour[:, 1], contour[:, 0], color='red',linewidth=1)
plt.axis('off') 
plt.show()
cv2.imwrite('bostmask5.jpg', full_img)
img[:,:,0][mask]=255
cv2.imwrite('submask4.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#mask=mask*255
#filename=
#cv2.imwrite(image_id, mask)

'''for image_id in testimages_list:
    img_id=delete_zero_bfstr(image_id.split('.')[0])
    y=950-int(float(img_id.split('_')[1])-LLcorner[1])
    print(y)'''