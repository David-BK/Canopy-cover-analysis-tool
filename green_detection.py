#!/usr/bin/env python
# coding: utf-8

# import packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import fnmatch
import sys
import getopt
from PIL import Image

# setup input and parameters
folder_path = ''
output = ''
try:
    opts, args = getopt.getopt(sys.argv[1:], "hf:o:c:u:e:s:a:v:l:p:n:", ["help", "folderpath", "output", "control", "huelower", "hueupper", "satlower", "satupper", "vallower", "valupper", "detectwhite", "noisereduction"])
except getopt.GetoptError:
    print('An unkown error occurred. Default usage: ')
    print('green_detection.py -f <folderpath>')
    sys.exit(2)


# set default parameters
control = False         # Whether to output control black and white images into <folderpath>/bw/ folder.
hue_lower = 36          # Recommended: 36. Lower limit of hue in HSV image for green detection.
hue_upper = 86          # Recommended: 86. Upper limit of hue in HSV image for green detection.
saturation_lower = 50   # Recommended: 50. Lower limit of saturation in HSV image for green detection.
saturation_upper = 255  # Recommended: 255. Upper limit of saturation in HSV image for green detection.
value_lower = 30        # Recommended: 30. Lower limit of value in HSV image for green detection.
value_upper = 255       # Recommended: 255. Lower limit of value in HSV image for green detection.
detect_white = False    # Recommended: True. Whether or not to include the detection of bright white colours, e.g. when to include white flowers.
noise_reduction = 100   # Recommended: 100. The amount of noise reduction to apply. In other words, the maximum area of detected objects to remove, either because of wrong detections or to remove small weeds and such.


for opt, arg in opts:
    if opt in ("-h", "--help"):
        print('Usage: \ngreendetection.py -f <folderpath>')
        print('\n')
        print('optional arguments: ')
        print('-o --output \tA string. The location and name of the output file. Default is <folderpath>/canopycover.txt.')
        print('\n')
        print('-c --control \tA boolean. Whether to output control black and white images into <folderpath>/bw/ folder. Default is False.')
        print('\n')
        print('-u --huelower \tAn integer. Lower limit of hue in HSV image for green detection. Default is 36.')
        print('\n')
        print('-e --hueupper \tAn integer. Upper limit of hue in HSV image for green detection. Default is 86.')
        print('\n')
        print('-s --satlower \tAn integer. Lower limit of saturation in HSV image for green detection. Default is 50.')
        print('\n')
        print('-a --satupper \tAn integer. Upper limit of saturation in HSV image for green detection. Default is 255.')
        print('\n')
        print('-v --vallower \tAn integer. Lower limit of value in HSV image for green detection. Default is 30.')
        print('\n')
        print('-l --valupper \tAn integer. Lower limit of value in HSV image for green detection. Default is 255.')
        print('\n')
        print('-p --detectwhite \tA boolean. Whether or not to include the detection of bright white colours, e.g. when there is a lot of leaf reflection or to include white flowers. Default is True.')
        print('\n')
        print('-n --noisereduction \tAn integer. How much noise reduction to apply, i.e. the maximum area of detected objects to remove. Default is 100')
        print('\n')
        sys.exit()
    elif opt in ("-f", "--fpath"):
        folder_path = arg
        folder_path = folder_path.replace(os.sep, '/')
    elif opt in ("-o", "--output"):
        output = arg
        output = output.replace(os.sep, '/')
    elif opt in ("-c", "--control"):
        control = arg.lower() == 'true'
    elif opt in ("-u", "--huelower"):
        hue_lower = int(arg)
    elif opt in ("-e", "--hueupper"):
        hue_upper = int(arg)
    elif opt in ("-s", "--satlower"):
        saturation_lower = int(arg)
    elif opt in ("-a", "--satupper"):
        saturation_upper = int(arg)
    elif opt in ("-v", "--vallower"):
        value_lower = int(arg)
    elif opt in ("-l", "--valupper"):
        value_upper = int(arg)
    elif opt in ("-p", "--detectwhite"):
        detect_white = arg.lower() == 'true'
    elif opt in ("-n", "--noisereduction"):
        noise_reduction = int(arg)

if not folder_path[-1] == '/':
    folder_path = folder_path + '/'
    
if len(output) == 0:
    output = folder_path + 'canopy_cover.txt'

# list image files
all_images = []
for file in os.listdir(folder_path):
    if fnmatch.fnmatch(file, '*.JPG') or fnmatch.fnmatch(file, '*.JPEG') or fnmatch.fnmatch(file, '*.HEIC'):
        all_images.append(file)

# set first image boolean
first_image = True

# loop over all images
for image_name in all_images:
    try:  
        # print message
        print('Processing image ' + image_name)
        
        # set image path
        image_path = folder_path + image_name

        # read image
        if fnmatch.fnmatch(image_name, '*.HEIC'):
            pil_img = Image.open(image_path).convert('RGB') 
            open_cv_img = np.array(pil_img) 
            # Convert RGB to BGR 
            img = open_cv_img[:, :, ::-1].copy() 
        else:
            img = cv2.imread(image_path)
        
        # convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # create bw image
        lower = np.array([hue_lower, saturation_lower, value_lower], dtype = "uint8")
        upper = np.array([hue_upper, saturation_upper, value_upper], dtype = "uint8")
        bw = cv2.inRange(hsv, lower, upper)
        
        if detect_white:
            thres_white = ([0, 0, 225], [255, 25, 255])
            lower_white = np.array(thres_white[0], dtype = "uint8")
            upper_white = np.array(thres_white[1], dtype = "uint8")
            bw_white = cv2.inRange(hsv, lower_white, upper_white)
            bw = cv2.bitwise_or(bw, bw_white)
        
        blur = cv2.GaussianBlur(bw, (3,3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Filter using contour area and remove small noise
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            if area < noise_reduction:
                cv2.drawContours(thresh, [c], -1, (0,0,0), -1)

        # Morph close and invert image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        bw = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # output control image
        if control:
            if not os.path.exists(folder_path + 'bw'):
                os.mkdir(folder_path + 'bw')
            cv2.imwrite(folder_path + 'bw' + '/' + image_name.replace('.JPG', '') + '_bw.JPG', bw)
        
        # count white pixels
        nwhite = np.sum(bw == 255)
        
        # calculate proportion green
        pgreen = np.round(nwhite / (bw.shape[0] * bw.shape[1]), 4)
        
        # create output file
        if first_image:
            if os.path.exists(output):
                os.remove(output)
            file = open(output, "x")
            file.write("image" + "\t" + "proportion_green" + "\n")
            file.write(image_name + "\t" + str(pgreen) + "\n")
            file.close()
            first_image = False
        else:
            file = open(output, "a")
            file.write(image_name + "\t" + str(pgreen) + "\n")
            file.close()
        
    except Exception as e:
        print("An error occurred:")
        print(e)
        print("\n")
