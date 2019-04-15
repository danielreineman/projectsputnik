import random
import numpy as np
import urllib.request
import cv2
from pyproj import Proj, transform
import time
P3857 = Proj(init='epsg:3857')
P4326 = Proj(init='epsg:4326')

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    filename=str(x)+'_'+str(y)+'.png'
    cv2.imwrite(filename, image)
    cv2.waitKey(0)
s = #size in meters
bbox = np.matrix([[-9743414.8579,4935480.3943],[-9695568.8612,4973554.3947]])
for n in range(1,501):
    x=random.randint(bbox[0,0]/100,bbox[1,0]/100)*100
    y=random.randint(bbox[0,1]/100,bbox[1,1]/100)*100
    box=np.matrix([[x,y],[x+100,y+100]])
    #burl = "http://gisservices.datadoors.net/i3_ArcGIS/wms/00a01748-be6f-430a-a16e-0a25f230a641?srs=EPSG:900913&request=getmap&layers=00a01748-be6f-430a-a16e-0a25f230a641&bbox="
    #eurl = "&width=300&height=300&format=image/jpg"
    burl = "http://www.valtus.com/views/wms?&userid=MytomtomEval&passwd=peter.berger4856&REQUEST=GetMap&LAYERS=HxIP_US_RGB&SRS=EPSG:3857&CRS=EPSG:3857&BBOX="
    eurl = "&WIDTH=6000&HEIGHT=6000&STYLES=&TRANSPARENT=true&FORMAT=image/png"
    url = burl + str(box[0,0]) + "," + str(box[0,1]) + "," + str(box[1,0]) + "," + str(box[1,1]) + eurl
    url_to_image(url)
    #time.sleep(0.01)
