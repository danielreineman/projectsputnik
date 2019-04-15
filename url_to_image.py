import numpy as np
import urllib.request
import cv2
import os
from pyproj import Proj, transform
import time
P3857 = Proj(init='epsg:3857')
P4326 = Proj(init='epsg:4326')
path = '/home/derose/Downloads/Indiana_hxgn/'
def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    filename=str(x[i])+'_'+str(y[l])+'.jpg'
    cv2.imwrite(os.path.join(path , filename), image)
    #cv2.imwrite(filename, image)
    cv2.waitKey(0)

#LLcorner=np.int32(np.float64(transform(P4326,P3857, -71.160873, 42.349358))/100)*100
SWcorner=np.int32((-9722675.3423,4953998.4587))
sizex=4120
sizey=4120
x=np.arange(SWcorner[0],SWcorner[0]+sizex,92)
y=np.arange(SWcorner[1],SWcorner[1]+sizey,92)
for i in range(0,len(x)):
    for l in range(0,len(y)):
        box=np.matrix([[x[i],y[l]],[x[i]+92,y[l]+92]])
        #burl = "http://gisservices.datadoors.net/i3_ArcGIS/wms/00a01748-be6f-430a-a16e-0a25f230a641?srs=EPSG:900913&request=getmap&layers=00a01748-be6f-430a-a16e-0a25f230a641&bbox="
        #eurl = "&width=300&height=300&format=image/jpg"
        burl = "http://www.valtus.com/views/wms?&userid=MytomtomEval&passwd=peter.berger4856&REQUEST=GetMap&LAYERS=HxIP_US_RGB&SRS=EPSG:3857&CRS=EPSG:3857&BBOX="
        eurl = "&WIDTH=300&HEIGHT=300&STYLES=&TRANSPARENT=true&FORMAT=image/png"
        url = burl + str(box[0,0]) + "," + str(box[0,1]) + "," + str(box[1,0]) + "," + str(box[1,1]) + eurl
        url_to_image(url)
        #time.sleep(0.01)
        