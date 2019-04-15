# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 11:38:37 2018

@author: grotton
"""

import urllib.request
import random

def downloader(image_url):
    file_name = random.randrange(1,10000)
    full_file_name = str(file_name) + '.png'
    urllib.request.urlretrieve(image_url,full_file_name)
