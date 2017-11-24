'''
Created on 23 Nov 2017

@author: Vanessa Brischi Olivatto
'''

# coding=utf-8

# Developed and tested with:
#
# os: Windows 10
# python version: 3.6.3
# opencv version: 3.3.1

# Notes:
# (1) Two different sets of parameters were used in Hough transform. First set is tuned to detect small circles. Second
# set is tuned to detect larger circles. Results from both transforms are concatenated to obtain final circle detection 
# (2) Pre-processing consists into opening and closing filtering to eliminate noise. Edge detection is added afterwards
#

import numpy as np
import cv2

import matplotlib.pyplot as plt
import sys


def Hough_circle_transform(path):

    # Read the image.
    image_ori = cv2.imread(path,0)
    

    # Apply erosion followed by dilation filter (Opening) in order to reduce noise (elliptical kernel).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(22, 22))   
    opening = cv2.morphologyEx(image_ori, cv2.MORPH_OPEN, kernel)    

    # Apply dilation followed by erosion filter (Closing) in order to reduce noise (elliptical kernel).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)    

    # Apply edge detection 
    image = cv2.Canny(closing, 0, 255)


    # Apply Hough Transform to small circles
    circles11 = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1.2, 60,
                            param1=50, param2=16, minRadius=4, maxRadius=10)
    # Apply Hough Transform to large circles
    circles12 = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1.4, 60,
                            param1=50, param2=60, minRadius=10, maxRadius=90)
    # Concatenate detected circles
    if circles11 is not None:
        circles = np.array([np.concatenate((circles11[0,:], circles12[0,:]), axis=0)])
    else:
        circles = circles12
     
    circles = np.uint16(np.around(circles))

    # Draw detected circles to the gray scale of the original image
    output = cv2.cvtColor(image_ori,cv2.COLOR_GRAY2BGR)
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(output,(i[0],i[1]),i[2],(255,0,0),2)
        
            
    bgr_img = cv2.imread(path)
    b,g,r = cv2.split(bgr_img)       # get b,g,r (bgr convention)
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
        
    plt.subplot(121),plt.imshow(rgb_img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(output)
    plt.title('Processed Image'), plt.xticks([]), plt.yticks([])
    plt.show()
        

if __name__ == '__main__':

    Hough_circle_transform(sys.argv[1])
        


