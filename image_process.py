#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 12:25:41 2016

@author: taofeng
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import imutils


def get_meta(d, j, p):
    x1=int(j[p]['annotations'][0]['x'])
    x2=int(j[p]['annotations'][1]['x'])
    y1=int(j[p]['annotations'][0]['y'])
    y2=int(j[p]['annotations'][1]['y'])
    if y1>y2:
        y=y2
        y2=y1
        y1=y
    if x1>x2:
        x=x2
        x2=x1
        x1=x
    return d+j[p]['filename'], x1, x2, y1, y2

def match(i, j):
    with open('alb_labels.json', 'r') as alb_position:
        alb_json = json.loads(alb_position.read())
        img_dir = 'data/train/ALB/'
        img_path1, x11, x12, y11, y12 = get_meta(img_dir, alb_json, i)
        img1 = cv2.imread(img_path1)
        img_crop = img1[y11:y12, x11:x12]
        img_path2, x21, x22, y21, y22 = get_meta(img_dir, alb_json, j)
        img2 = cv2.imread(img_path2)
        (img1Height, img1Width) = img_crop.shape[:2]
        result = cv2.matchTemplate(img2, img_crop, cv2.TM_CCOEFF)
        (_, _, minLoc, maxLoc) = cv2.minMaxLoc(result)
        # the puzzle image
        topLeft = maxLoc
        botRight = (topLeft[0] + img1Width, topLeft[1] + img1Height)
        roi = img2[topLeft[1]:botRight[1], topLeft[0]:botRight[0]]
        # construct a darkened transparent 'layer' to darken everything
        # in the puzzle except for waldo
        mask = np.zeros(img2.shape, dtype = "uint8")
        img2 = cv2.addWeighted(img2, 0.25, mask, 0.75, 0)
        
        # put the original waldo back in the image so that he is
        # 'brighter' than the rest of the image
        img2[topLeft[1]:botRight[1], topLeft[0]:botRight[0]] = roi
        # display the images
        cv2.imshow("Img2", imutils.resize(img2, height = 650))
        cv2.imshow("Img1", img_crop)
        cv2.waitKey(0)


def match_boat():
    img1_path = 'data/train/NOF/img_00008.jpg'
    img2_path = 'data/train/NOF/img_00011.jpg'
    img1 = cv2.imread(img1_path,0)          # queryImage
    img2 = cv2.imread(img2_path,0) # trainImage
    
    # Initiate SIFT detector
    sift = cv2.SIFT()
     
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    
    # Apply ratio test
    good = []
    for m,n in matches:
       if m.distance < 0.75*n.distance:
          good.append([m])
    
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)
    
    plt.imshow(img3),plt.show()
    
match_boat()