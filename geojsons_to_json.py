#%%
from __future__ import division 
import json
import ijson
from numpy import *
from numpy import array as ar
from pyproj import Proj, transform
from math import sqrt
from sys import getsizeof as gs
from sparse_list import *
from sparse_list import SparseList as sl
P3857 = Proj(init='epsg:3857')
P4326 = Proj(init='epsg:4326')
import random
import urllib.request
import cv2
import matplotlib.pyplot as plt
from time import sleep
#%%
def Centroid(poly): #polygon goes in, centroid comes out
    center=ar([mean(poly[:,0]),mean(poly[:,1])])
    x1, y1 = center
    Areas = []
    TriCentroids = []
    for i in range(len(poly)-1):       
        x2, y2 = poly[i,:]
        x3, y3 = poly[i+1,:]
        Area = x1*y2 + x2*y3 + x3*y1 - x2*y1 - x3*y2 - x1*y3
        TriCentroid = ar([mean([x1,x2,x3]),mean([y1,y2,y3])])
        Areas.append(Area)
        TriCentroids.append(TriCentroid)
    return ar([sum(multiply(ar(Areas),ar(TriCentroids)[:,0])),sum(multiply(ar(Areas),ar(TriCentroids)[:,1]))])/sum(Areas)

def PolyArea(poly):
    center=ar([mean(poly[:,0]),mean(poly[:,1])])
    x1, y1 = center
    Areas = []
    for i in range(len(poly)-1):       
        x2, y2 = poly[i,:]
        x3, y3 = poly[i+1,:]
        Area = x1*y2 + x2*y3 + x3*y1 - x2*y1 - x3*y2 - x1*y3
        Areas.append(Area)
    return abs(round(float(sum(ar(Areas)))))

from typing import Iterable
#from math import floor, sqrt
from math import sqrt
def szudzik_pairing(index1: int, index2: int):
    if index1 > index2:
        return index1 ** 2 + index2
    else:
        return index2 ** 2 + index2 + index1
sp = szudzik_pairing

def szudzik_unpairing(index: int):
    shell = floor(sqrt(index))
    if index - shell ** 2 < shell:
        return [shell, index - shell ** 2]
    else:
        return [index - shell ** 2 - shell, shell]
su = szudzik_unpairing

def line(p1, p2): #input two points, outputs 3 variables to be used by intersection
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2): #input two lines in format given by line, outputs intersection of lines
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def PolyCrop(poly): #crops polygons to bounding box 0 < x < 100, 0 < x < 100
    poly = poly[0:-1] #removes last point which is a duplicate of the first. easier for computation, and i add it back later
#    plt.figure()
#    plt.plot(poly[:,0],poly[:,1])
    should_restart0 = True
    while should_restart0: #some idiot recommended that i use a while loop
        should_restart0 = False     
        for l in range(len(poly)): #for every point in poly
            #print('starting right boundary')
            #print('l =',l)
            L1 = line((100,0),(100,100)) #bounding box edge
            poly=round_(poly*10**6)/10**6
            if poly[l-1][0] > 100: #is this point outside x=100?
                #print(poly[l-1])
                if poly[l-2][0] < 100 and poly[l][0] < 100: #are both of its neighbors within x=100?            
                    L2 = line(poly[l-1],poly[l-2]) #line connecting point to its previous neighbor
                    L3 = line(poly[l-1],poly[l]) #line connecting point to its next neighbor
                    P1 = ar([intersection(L1, L2)]) #intersection of L1 and L2
                    P2 = ar([intersection(L1, L3)]) #intersection of L1 and L3
                    if l==0:
                        poly = vstack((poly[:len(poly)-1],P1,P2)) #replace point that was outside x=100 with P1 and P2 (if l=0)
                    else:    
                        poly = vstack((poly[:l-1],P1,P2,poly[l:len(poly)])) #replace point that was outside x=100 with P1 and P2 (if l=/=0)
                    #print('1 point right')
                    should_restart0 = True #start over because we just shifted poly around and the indices are different
                    break
                if poly[l-2][0] >= 100 and poly[l][0] >= 100: #are this points' neighbors outside x=100 or on x=100?
                    poly = delete(poly, (l-1), axis=0) # We get rid of all points not connected to a point within x=100
                    #print('3+ points right')
                    should_restart0 = True #deleted a point so restart
                    break
                if poly[l][0] < 100: #is the next point within x=100?
                    L2 = line(poly[l-1],poly[l]) #line connecting this point to its next neighbor
                    poly[l-1]=ar([intersection(L1, L2)]) #shift this point to the intersection of x=100 and L2
                    #print('2 points right forward')
                    should_restart0 = True #restarting for safety, might try removing
                    break
                elif poly[l-2][0] < 100: #is the previous point within x=100?
                    L2 = line(poly[l-1],poly[l-2]) #line connecting this point to its previous neighbor
                    poly[l-1]=ar([intersection(L1, L2)]) #shift this point to the intersection of x=100 and L2
                    #print('2 points right backward')
                    should_restart0 = True #restarting for safety, might try removing
                    break
    should_restart1 = True
    while should_restart1:
        should_restart1 = False
        for l in range(len(poly)):
            L1 = line((0,0),(0,100))
            poly=round_(poly*10**6)/10**6
            if poly[l-1][0] < 0:
                if poly[l-2][0] > 0 and poly[l][0] > 0:                
                    L2 = line(poly[l-1],poly[l-2])
                    L3 = line(poly[l-1],poly[l])
                    P1 = ar([intersection(L1, L2)])
                    P2 = ar([intersection(L1, L3)])
                    if l==0:
                        poly = vstack((poly[:len(poly)-1],P1,P2))
                    else:    
                        poly = vstack((poly[:l-1],P1,P2,poly[l:len(poly)]))
                    should_restart1 = True
                    break
                if poly[l-2][0] <= 0 and poly[l][0] <= 0:
                    poly = delete(poly, (l-1), axis=0)
                    #print('deleted point left of box')
                    should_restart1 = True
                    break
                 
                if poly[l][0] > 0:
                    L2 = line(poly[l-1],poly[l])
                    poly[l-1]=ar([intersection(L1, L2)])
                    #print('shifted point to left')
                    should_restart1 = True
                    break
                elif poly[l-2][0] > 0:
                    L2 = line(poly[l-1],poly[l-2])
                    poly[l-1]=ar([intersection(L1, L2)])
                    #print('shifted point to left')
                    should_restart1 = True
                    break
    should_restart2 = True
    while should_restart2:
        should_restart2 = False
        #print('starting upper boundary')
        for l in range(len(poly)):
            #print('l =',l)
            L1 = line((0,100),(100,100))
            poly=round_(poly*10**6)/10**6
            if poly[l-1][1] > 100:
                #print(poly[l-1])
                if poly[l-2][1] < 100 and poly[l][1] < 100:                
                    L2 = line(poly[l-1],poly[l-2])
                    L3 = line(poly[l-1],poly[l])
                    P1 = ar([intersection(L1, L2)])
                    P2 = ar([intersection(L1, L3)])
                    if l==0:
                        poly = vstack((poly[:len(poly)-1],P1,P2))
                    else:    
                        poly = vstack((poly[:l-1],P1,P2,poly[l:len(poly)]))
                    #print('1 point above')
                    should_restart2 = True
                    break
                if poly[l-2][1] >= 100 and poly[l][1] >= 100:
                    poly = delete(poly, (l-1), axis=0)
                    #print('3+ points above')
                    should_restart2 = True
                    break              
                if poly[l][1] < 100:
                    L2 = line(poly[l-1],poly[l])
                    poly[l-1]=ar([intersection(L1, L2)])
                    #print('2 points above forward')
                    should_restart2 = True
                    break
                elif poly[l-2][1] < 100:
                    L2 = line(poly[l-1],poly[l-2])
                    poly[l-1]=ar([intersection(L1, L2)])
                    #print('2 points above backward')
                    #print(poly[l-1])
                    should_restart2 = True
                    break
    should_restart3 = True
    while should_restart3:
        should_restart3 = False
        for l in range(len(poly)):
            L1 = line((0,0),(100,0)) 
            poly=round_(poly*10**6)/10**6
            if poly[l-1][1] < 0:
                if poly[l-2][1] > 0 and poly[l][1] > 0:                
                    L2 = line(poly[l-1],poly[l-2])
                    L3 = line(poly[l-1],poly[l])
                    P1 = ar([intersection(L1, L2)])
                    P2 = ar([intersection(L1, L3)])
                    if l==0:
                        poly = vstack((poly[:len(poly)-1],P1,P2))
                    else:    
                        poly = vstack((poly[:l-1],P1,P2,poly[l:len(poly)]))
                    should_restart3 = True
                    break
                if poly[l-2][1] <= 0 and poly[l][1] <= 0:
                    poly = delete(poly, (l-1), axis=0)
                    #print('deleted point below box')
                    should_restart3 = True
                    break                           
                if poly[l][1] > 0:
                    L2 = line(poly[l-1],poly[l])
                    poly[l-1]=ar([intersection(L1, L2)])
                    #print('shifted point to bottom')
                    should_restart3 = True
                    break
                elif poly[l-2][1] > 0:
                    L2 = line(poly[l-1],poly[l-2])
                    poly[l-1]=ar([intersection(L1, L2)])
                    #print('shifted point to bottom')
                    should_restart3 = True
                    break
    should_restart4 = True
    while should_restart4:
        should_restart4 = False
        for l in range(len(poly)-1): #deletes points that duplicate each other. Never benchmarked how often it happens
            poly=round_(poly*10**6)/10**6
            if all(poly[l] == poly[l+1]):
                poly = delete(poly, (l+1), axis=0)
                #print('deleted duplicate point')
                should_restart4 = True
                break
    poly=vstack((poly,poly[0])) #adds first point back to the end
    return poly
    #plt.figure()
    #plt.plot(poly[:,0],poly[:,1])
def loc_to_image(loc): #inpiut the index of the tile, output returns an image. Will most likely change 
    #to save image to a specified directory with imageid = index of tile rather than return image    
    x,y=loc*100+SWcorner
    box=ar([[x,y],[x+100,y+100]])
    burl = "http://gisservices.datadoors.net/i3_ArcGIS/wms/00a01748-be6f-430a-a16e-0a25f230a641?srs=EPSG:3857&request=getmap&layers=00a01748-be6f-430a-a16e-0a25f230a641&bbox="
    eurl = "&width=300&height=300&format=image/jpg"
    url = burl + str(box[0,0]) + "," + str(box[0,1]) + "," + str(box[1,0]) + "," + str(box[1,1]) + eurl
    resp = urllib.request.urlopen(url)
    image = asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    imname=path + folder + str(k)+'.png'
    cv2.imwrite(imname, image)
    return image
    #sleep(.01)
    cv2.waitKey(0)

def merc2pxl(pc): #converts polygon in mercator coordinates to pixels. Does not flatten for sake of bbbox def
    pp=empty_like(pc)
    pp[:,0]=floor(3*(100-pc[:,1]))
    pp[:,1]=floor(3*pc[:,0])
    pp=int16(pp)
    return pp

def bbbox(pp): #input polygon in pixel format, output dict with keys x, y, width, height
    bbbox=[min(pp[:,1]),min(pp[:,0]),max(pp[:,1])-min(pp[:,1]),max(pp[:,0])-min(pp[:,0])]
    return bbbox
#%%
states_buildings={
'Alabama':      2460404,'Arizona':      2555395,#'Alaska':       110746,
'Arkansas':     1508657,'California':   10988525,'Colorado':     2080808,
'Connecticut':  1190229,'Delaware':     345907,'DistrictofColumbia':58329,
'Florida':      6903772,'Georgia':      3873560,#'Hawaii':       252891,
'Idaho':        883594,'Illinois':     4855794,'Indiana':      3268325,
'Iowa':         2035688,'Kansas':       1596495,'Kentucky':     2384214,
'Louisiana':    2057368,'Maine':        752054,'Maryland':     1622849,
'Massachusetts':2033018,'Michigan':     4900472,'Minnesota':    2815784,
'Mississippi':  1495864,'Missouri':     3141265,'Montana':      762288,
'Nebraska':     1158081,'Nevada':       932025,'New Hampshire':563487,
'New Jersey':   2480332,'New Mexico':   1011373,'New York':     4844438,
'North Carolina':4561262,'North Dakota': 559161,'Ohio':         5449419,
'Oklahoma':     2091131,'Oregon':       1809555,'Pennsylvania': 4850273,
'Rhode Island': 366779,'South Carolina':2180513,'South Dakota': 649737,
'Tennessee':    3002503,'Texas':        9891540,'Utah':         1004734,
'Vermont':      345911,'Virginia':     3057019,'Washington':   2993361,
'West Virginia':1020031,'Wisconsin':    3054452,'Wyoming':      380772}

val_json_small=json.load(open('/home/derose/osmc/data/val/annotation-small.json'))
categories=val_json_small['categories']
info=val_json_small['info']
states=list(states_buildings) #converts states dictionary into a list over whose items i can iterate
path='/media/derose/External/' #path to data
SWcorner=ar([-13900000,2900000]) #Important: this is the origin of our 'Murica-based coordinate system
#country_tiles=[] #empty list for all the tiles
#country_buildings=[] #empty list of tiles which are lists of buildings
#%%
tiles=[]
buildings=[]
train_annotations=list([])
train_images=list([])
val_annotations=list([])
val_images=list([])
#for state in states:   #iterates over all states. When ready, uncommend and indent everything below EXCEPT country_buildings = country_buildings[0] and country_tiles = country_tiles[0]
    #state_tiles=[] #empty list of tiles for just the working state
    #state_buildings=[] #empty list of tiles which are lists of buildings only for the working state
X=sl(65000) #sparse list for 65000 vertical strips of land
for i in range(0,len(X)):
    X[i]=sl(34000) #sparse list for 34000 tiles, each within one vertical strip
filename = path + 'states/' + 'Washington' + '.geojson' #file we're reading. When doing whole country,c hange states[index] to just state           
TL=len(tiles)
with open(filename) as f: #opens file
    objects = ijson.items(f, 'features.item.geometry.coordinates.item') #generator
    for polygon in objects: #streams polygons from generator one at a time
        poly64=float64(ar(polygon)) #converts decimal to float64 array
        poly64[:,0],poly64[:,1] = transform(P4326,P3857, poly64[:,0],poly64[:,1]) #long,lat to mercator  
        centroid = Centroid(poly64) #finds centroid
        loc=int32(floor((centroid-SWcorner)/100)) #location in sparse list of sparse lists comes from centroid
        poly16=float16(poly64-(SWcorner+loc*100)) #transforms to new coordinate system with respect to LLcorner of tile
        #converts to float16 because 64 bit precision is not required for such distances.
        try:
            X[loc[0]][loc[1]].append(poly16) #try to append, only works if something is there already
        except Exception: #if you can't do it
            X[loc[0]][loc[1]]=[] #put nothing there
            X[loc[0]][loc[1]].append(poly16) #then do it
        r=random.randrange(0,1000) #random number to select sample of buildings
        if r <= 2: #3 in 1000 chance
            #state_tiles.append(loc) #save that location to state_tiles for later use
            tiles.append(loc)
            #image=loc_to_image(loc)
        if not any(centroid[0] > poly64[:,0]) and any(centroid[0] < poly64[:,0]) and any(centroid[1] > poly64[:,1]) and any(centroid[1] < poly64[:,1]):
            print(centroid) #these 4 lines determine if a centroid is outside of its polygon's bbox
            print(poly64) #I'm awesome and wrote my own definition of centroid which always works
            break #so these 4 lines are justin case
#country_tiles.append(state_tiles) #done iterating over state so we dump tiles into country before clearing state_tiles
#%%
#for k in range(TL,len(tiles)): #now we go into state_tiles and do stuff   
for k in range(0,10):    
    #print('k =',k)
    tile_buildings=[] #empty slate to put buildings into
    r = random.random()
    if r < 0.2:
        val = True
        folder = 'val/'
    else:
        val = False
        folder = 'train/'
    loc = tiles[k] #get location out. realizing now that i could have done "for loc in state tiles"
    img=loc_to_image(loc)
    bbox=ar([[SWcorner+loc*100], #remember that we're dealing with a tile-local coordinate system and have to reference country
             [SWcorner+loc*100+100]])    
    building_number = 0
    for n in range(-1,2): #looking in 3x3 grid with our working one in the middle, to find buildings that are partially in our tile
        #print('n =',n)
        for m in range(-1,2):           
            #print('m =',m)
            try:
                length = len(X[loc[0]+n][loc[1]+m]) #length of tile will be number of buildings there. doesn't work if nothing is there
                #print('length =', length)
                for l in range(0,length):
                    #print('l =', l)
                    poly=X[loc[0]+n][loc[1]+m][l]+SWcorner+ar([loc[0]+n,loc[1]+m])*100 #transforming to country coordinates to compare
                    a=logical_and(greater_equal(poly,bbox[0]), less_equal(poly,bbox[1])) #looking for points within our tile
                    b=logical_and(a[:,0],a[:,1]) #gotta be both x and y for a point
                    if any(b):
                        #print(any(b))                           
                        polygood = poly-bbox[0] #this is a good poly because it's in our tile
                        polycropped = PolyCrop(polygood) #crop to boundary of tile/image
                        polypixel = merc2pxl(polycropped) #convert to pixels (100x100 meter to 300x300 pixel)
                        tile_buildings.append(polypixel) #puts in tile_buildings
                        area=PolyArea(polypixel)
                        box=bbbox(polypixel) #returns coco-formatted dict for bbox. do before flattening
                        polypixel=polypixel.flatten() #flattens polypixel for coco format
                        '''r = random.random()
                        if val:
                            val_annotations.append({
                            'area': abs(round(float(area))), 
                            'bbox': [float(i) for i in box],
                            'category_id': 100,
                            'id': sp(k,building_number),
                            'image_id': k,
                            'iscrowd': 0,
                            'segmentation': [[float(i) for i in polypixel.tolist()]]})
                            val_images.append({
                            'filename': k + '.jpg',
                            'height': int(300),
                            'id': k,
                            'width': int(300)})
                        else:
                            train_annotations.append({
                            'area': abs(round(float(area))), 
                            'bbox': [float(i) for i in box],
                            'category_id': 100,
                            'id': sp(k,building_number),
                            'image_id': k,
                            'iscrowd': 0,
                            'segmentation': [[float(i) for i in polypixel.tolist()]]})
                            train_images.append({
                            'filename': k + '.jpg',
                            'height': int(300),
                            'id': k,
                            'width': int(300)})'''
                        building_number = building_number + 1 
                        #plt.xlim(-20, 120)
                        #plt.ylim(-20, 120)
                        #plt.plot(polygood[:,0],polygood[:,1])
                        #plt.xlim(-20, 120)
                        #plt.ylim(-20, 120)
                        #plt.plot(polycropped[:,0],polycropped[:,1])
                        #image id will simply be the index in country_tiles corresponding to that image
                        #building id will be szudzik pair of these indices: country_buildings[tile][building]
                        #example: sp(tile,building)
                        #example: sp(20,5) = 405
                        #example: su(405) = 20, 5
                        
            except: #no buildings in tile? move on
                pass
    plt.figure()
    plt.imshow(img)
    for a in range(0,len(tile_buildings)):
        plt.plot(tile_buildings[a][:,1],tile_buildings[a][:,0])
    #state_buildings.append(tile_buildings) #puts into state_buildings
    
#country_buildings.append(state_buildings) #puts into country_buildings
#country_buildings = country_buildings[0] #peels off extraneous layer. do not include in state indent
#country_tiles = country_tiles[0] #peels off extraneous layer. do not include in state indent

json.dump({
    'annotations':train_annotations,
    'categories':categories,
    'images':train_images,
    'info':info}, open('train.json', 'w'))
json.dump({
    'annotations':val_annotations,
    'categories':categories,
    'images':val_images,
    'info':info}, open('val.json', 'w'))
    
