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
    opts, args = getopt.getopt(sys.argv[1:], "hf:o:c:u:e:s:a:v:l:p:n:k:w:i:r:d:u:t:b:x:y:", ["help", "folderpath", "output", "control", "huelower", "hueupper", "satlower", "satupper", "vallower", "valupper", "detectwhite", "noisereduction", "kernelsize", "brightnessthreslow", "brightnessthreshigh", "erode1", "dilate", "erode2", "offtop", "offbot", "offleft", "offright"])
except getopt.GetoptError:
    print('An unkown error occurred. Default usage: ')
    print('canopy_cover.py -f <folderpath>')
    sys.exit(2)

for opt, arg in opts:
    if opt in ("-h", "--help"):
        print('Usage: \ncrop_frame_image.py -f <folderpath>')
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
        print('-p --detectwhite \tA boolean. Whether or not to include the detection of bright white colours, e.g. when there is a lot of leaf reflection or when to include white flowers. Default is True.')
        print('\n')
        print('-n --noisereduction \tAn integer. How much noise reduction to apply, i.e. the maximum area of detected objects to remove. Default is 100.')
        print('\n')
        print('-k --kernelsize \tAn integer. Size of the kernel used for erosion and dilation modifications to detect the frame. Default is the rounded integer of image width divided by 4000 (10 for an image of size 3000x4000).')
        print('\n')
        print('-w --brightnessthreslow \tAn integer. Lower threshold of the colour detection for the frame. Default is 130.')
        print('\n')
        print('-i --brightnessthreshigh \tAn integer. Upper threshold of the colour detection for the frame. Default is 255.')
        print('\n')
        print('-r --erode1 \t\tAn integer. Number of iterations of the first erosion operation on the image to detect the frame. Default is 4.')
        print('\n')
        print('-d --dilate \t\tAn integer. Number of iterations of the dilation operation on the image to detect the frame. Default is 4.')
        print('\n')
        print('-u --erode2 \t\tAn integer. Number of iterations of the second erosion operation on the image to detect the frame. Default is 1.')
        print('\n')
        print('-t --offtop \t\tAn integer. Number of pixels to be offset at the top of the picture (in case a part of the frame is visible in every cropped image). Default is the rounded integer of image width divided by 60 (67 for an image of size 3000x4000)')
        print('\n')
        print('-b --offbot \t\tAn Integer. Number of pixels to be offset at the bottom of the picture. Default is the rounded integer of image width divided by 60 (67 for an image of size 3000x4000).')
        print('\n')
        print('-x --offleft \t\tAn Integer. Number of pixels to be offset at the left of the picture. Default is the rounded integer of image width divided by 60 (67 for an image of size 3000x4000).')
        print('\n')
        print('-y --offright \t\tAn Integer. Number of pixels to be offset at the right of the picture. Default is the rounded integer of image width divided by 60 (67 for an image of size 3000x4000).')
        print('\n')
        sys.exit()
    elif opt in ("-f", "--fpath"):
        folder_path = arg
        folder_path = folder_path.replace(os.sep, '/')

if not folder_path[-1] == '/':
    folder_path = folder_path + '/'

# list image files
all_images = []
for file in os.listdir(folder_path):
    if fnmatch.fnmatch(file, '*.JPG') or fnmatch.fnmatch(file, '*.JPEG'):
        all_images.append(file)

# set first image
image_name = all_images[0]

# set image path
image_path = folder_path + image_name

# read image
img = cv2.imread(image_path)

image_dims = img.shape

# set default parameters
control = False                                     # Whether to output control black and white images into <folderpath>/bw/ folder.
hue_lower = 36                                      # Recommended: 36. Lower limit of hue in HSV image for green detection.
hue_upper = 86                                      # Recommended: 86. Upper limit of hue in HSV image for green detection.
saturation_lower = 50                               # Recommended: 50. Lower limit of saturation in HSV image for green detection.
saturation_upper = 255                              # Recommended: 255. Upper limit of saturation in HSV image for green detection.
value_lower = 30                                    # Recommended: 30. Lower limit of value in HSV image for green detection.
value_upper = 255                                   # Recommended: 255. Lower limit of value in HSV image for green detection.
detect_white = True                                 # Recommended: False. Whether or not to include the detection of bright white colours, e.g. when to include white flowers.
noise_reduction = 100                               # Recommended: 100. The amount of noise reduction to apply. In other words, the maximum area of detected objects to remove, either because of wrong detections or to remove small weeds and such.
kernel_size = int(np.round(image_dims[1] / 400))    # Recommended: 10 for image dims of 3000x4000. Size of the kernel used for erosion and dilation modifications to detect the frame
brightness_thres_low = 130                          # Recommended: 130. Lower threshold of the brightness detection for the frame
brightness_thres_high = 255                         # Recommended: 255. Upper threshold of the brightness detection for the frame
erode_it_1 = 4                                      # Recommended: 6. Number of iterations of the first erosion operation on the image to detect the frame. For noisy images (frame not much brighter than rest of image), a low value is recommended.
dilate_it_1 = 4                                     # Recommended: 25. Number of iterations of the first dilation operation on the image to detect the frame. For noisy images (frame not much brighter than rest of image), a low value is recommended.
erode_it_2 = 1                                      # Recommended: 20. Number of iterations of the second erosion operation on the image to detect the frame. For noisy images (frame not much brighter than rest of image), a low value is recommended.
offset_top = int(np.round(image_dims[1] / 60))      # Recommended: 0. Number of pixels to be offset at the top of the picture (in case a part of the frame is visible in every cropped image)
offset_bottom = int(np.round(image_dims[1] / 60))   # Recommended: -100 for image dims of 3000x4000. Number of pixels to be offset at the bottom of the picture
offset_left = int(np.round(image_dims[1] / 60))     # Recommended: 10 for image dims of 3000x4000. Number of pixels to be offset at the left of the picture
offset_right = int(np.round(image_dims[1] / 60))    # Recommended: -100 for image dims of 3000x4000. Number of pixels to be offset at the right of the picture

for opt, arg in opts:
    if opt in ("-o", "--output"):
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
    elif opt in ("-k", "--kernelsize"):
        kernel_size = int(arg)
    elif opt in ("-w", "--brightnessthreslow"):
        brightness_thres_low = int(arg)
    elif opt in ("-i", "--brightnessthreshigh"):
        brightness_thres_high = int(arg)
    elif opt in ("-r", "--erode1"):
        erode_it_1 = int(arg)
    elif opt in ("-d", "--dilate"):
        dilate_it_1 = int(arg)
    elif opt in ("-u", "--erode2"):
        erode_it_2 = int(arg)
    elif opt in ("-t", "--offtop"):
        offset_top = int(arg)
    elif opt in ("-b", "--offbot"):
        offset_bottom = int(arg)
    elif opt in ("-x", "--offleft"):
        offset_left = int(arg)
    elif opt in ("-y", "--offright"):
        offset_right = int(arg)

if not folder_path[-1] == '/':
    folder_path = folder_path + '/'
    
if len(output) == 0:
    output = folder_path + 'canopy_cover.txt'

crop_frame_image_call = "python crop_frame_image.py -f \"" + folder_path + "\" -k " + str(kernel_size) + " -w " + str(brightness_thres_low) + " -i " + str(brightness_thres_high) + " -r " + str(erode_it_1) + " -d " + str(dilate_it_1) + " -u " + str(erode_it_2) + " -t " + str(offset_top) + " -b " + str(offset_bottom) + " -x " + str(offset_left) + " -y " + str(offset_right)

green_detection_call = "python green_detection.py -f \"" + folder_path + "cropped/" + "\" -o \"" + output + "\" -c " + str(control) + " -u " + str(hue_lower) + " -e " + str(hue_upper) + " -s " + str(saturation_lower) + " -a " + str(saturation_upper) + " -v " + str(value_lower) + " -l " + str(value_upper) + " -p " + str(detect_white) + " -n " + str(noise_reduction)

# Run crop_frame_image.py
os.system(crop_frame_image_call)

# Run green_detection.py
os.system(green_detection_call)
