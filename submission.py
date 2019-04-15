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

with open('/home/derose/Downloads/indiana_bing.json','r') as f:
    prediction_json=json.load(f)

testimages_dir='/home/derose/Downloads/Indiana_bing'
#testimages_dir='/home/derose/osmc/data/test_images'
testimages_list=os.listdir(testimages_dir)
output_dir='/home/derose/Downloads/Indiana_bing_mask'
    #image_id=random.choice(testimages_list)
for image_id in testimages_list:#[0:401]:
    #image_id='000000009271.jpg'
    img_filepath=os.path.join(testimages_dir,image_id)
    img=mpimg.imread(img_filepath)
    img_real=mpimg.imread(img_filepath)
    mask=np.zeros(img.shape)[:,:,0]
    img_id=delete_zero_bfstr(image_id.split('.')[0])
    img_annlist=find_id_ann(prediction_json,img_id)
    for ann in img_annlist:
        m=cocomask.decode(ann['segmentation'])
        mask+=m       
        M = cv2.moments(m)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        size=sum(sum(m))
        #print(size)
        #m=m*255
        house=np.matrix([cX,cY]).astype('float')
        #filename=str(round(ann['score']*100)/100)+image_id
        #cv2.imwrite(filename, m)'''
    mask=mask>0
    contours = measure.find_contours(mask, 0.5)
    img.flags.writeable=True
    img[:,:,0][mask]=255
    '''plt.figure()
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
    plt.show()'''
    for ann in img_annlist:
        m=cocomask.decode(ann['segmentation'])     
        M = cv2.moments(m)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if sum(sum(m)) >= 50:
           cv2.circle(img, (cX, cY), 4, (0, 255, 255), -1)
           #cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) 
    #plt.figure()
    #plt.imshow(img)
    cv2.imwrite(os.path.join(output_dir , image_id), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    #cv2.imwrite(image_id, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#mask=mask*255
#filename=
#cv2.imwrite(image_id, mask)