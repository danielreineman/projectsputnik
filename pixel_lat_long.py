#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 14:14:42 2018

@author: derose
"""
import matplotlib.image as img
import numpy
def array2cartesian(pos, arr):
    pos=pos.astype('float')
    x1 = pos[0,1]/numpy.shape(arr)[1]
    x2 = -pos[0,0]/numpy.shape(arr)[0]+1
    pos=numpy.matrix([x1,x2])
    return pos
#house=numpy.matrix([450,620]).astype('float')


'''box = numpy.matrix([[-9824666.76,5095907.58],
     [-9822666.76,5097907.58]])'''
#image = img.imread('test.png')

scale = numpy.matrix([box[1,0]-box[0,0],box[1,1]-box[0,1]])

house = numpy.multiply(scale,array2cartesian(house, m))+box[0,:]

from pyproj import Proj, transform

P3857 = Proj(init='epsg:3857')
P4326 = Proj(init='epsg:4326')

house[:,0], house[:,1] = transform(P3857, P4326, house[:,0], house[:,1])

#box[:,0], box[:,1] = transform(P4326, P3857, 35, 60)