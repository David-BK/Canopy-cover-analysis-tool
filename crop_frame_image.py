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
from itertools import combinations

# setup input and parameters
folder_path = ''
try:
    opts, args = getopt.getopt(sys.argv[1:], "hf:k:w:i:r:d:u:t:b:x:y:", ["help", "folderpath", "kernelsize", "brightnesshreslow", "brightnessthreshigh", "erode1", "dilate", "erode2", "offtop", "offbot", "offleft", "offright"])
except getopt.GetoptError:
    print('An unkown error occurred. Default usage: ')
    print('crop_frame_image.py -f <folderpath>')
    sys.exit(2)

for opt, arg in opts:
    if opt in ("-h", "--help"):
        print('Usage: \ncrop_frame_image.py -f <folderpath>')
        print('\n')
        print('optional arguments: ')
        print('-k --kernelsize \tAn integer. Size of the kernel used for erosion and dilation modifications to detect the frame. Default is the rounded integer of image width divided by 400 (10 for an image of size 3000x4000).')
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
    if fnmatch.fnmatch(file, '*.JPG') or fnmatch.fnmatch(file, '*.JPEG') or fnmatch.fnmatch(image_name, '*.HEIC'):
        all_images.append(file)

# set first image
image_name = all_images[0]

# set image path
image_path = folder_path + image_name

# read image
img = cv2.imread(image_path)

image_dims = img.shape

# set default parameters
kernel_size = int(np.round(image_dims[1] / 400))    # Recommended: 10 for image dims of 3000x4000. Size of the kernel used for erosion and dilation modifications to detect the frame
brightness_thres_low = 130                          # Recommended: 130. Lower threshold of the brightness detection for the frame
brightness_thres_high = 255                         # Recommended: 255. Upper threshold of the brightness detection for the frame
erode_it_1 = 4                                      # Recommended: 6. Number of iterations of the first erosion operation on the image to detect the frame. For noisy images (frame not much brighter than rest of image), a low value is recommended.
dilate_it_1 = 4                                     # Recommended: 25. Number of iterations of the first dilation operation on the image to detect the frame. For noisy images (frame not much brighter than rest of image), a low value is recommended.
erode_it_2 = 1                                      # Recommended: 20. Number of iterations of the second erosion operation on the image to detect the frame. For noisy images (frame not much brighter than rest of image), a low value is recommended.
offset_top = int(np.round(image_dims[1] / 60))      # Recommended: 50. Number of pixels to be offset at the top of the picture (in case a part of the frame is visible in every cropped image)
offset_bottom = int(np.round(image_dims[1] / 60))   # Recommended: 50 for image dims of 3000x4000. Number of pixels to be offset at the bottom of the picture
offset_left = int(np.round(image_dims[1] / 60))     # Recommended: 50 for image dims of 3000x4000. Number of pixels to be offset at the left of the picture
offset_right = int(np.round(image_dims[1] / 60))    # Recommended: 50 for image dims of 3000x4000. Number of pixels to be offset at the right of the picture

for opt, arg in opts:
    if opt in ("-s", "--square"):
        square = arg.lower() == 'true'
    elif opt in ("-e", "--expansion"):
        expansion = int(arg)
    elif opt in ("-k", "--kernelsize"):
        kernel_size = int(arg)
    elif opt in ("-w", "--brightnessthreslow"):
        brightness_thres_low = int(arg)
    elif opt in ("-i", "--brightnesshreshigh"):
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

def search_4points(points,direction,w,h):
    '''
    Written by Ruokai Lin, 20-10-2022
    
    This function seperate the points into four parts
    SE means seperated points
    but it is redundant (called four times in a loop), can be simplified 
    '''
    temp = []
    if direction == 1:
        se = [p for p in points if 0<p[0]< w/4 and 0<p[1]<h/4]
    elif direction == 2:
        se = [p for p in points if 0<p[0]< w/4 and p[1]>h*3/4]
    elif direction == 3:
        se = [p for p in points if p[0]> w*3/4 and h/4>p[1]>0]
    elif direction == 4:
        se = [p for p in points if p[0]> w/4 and p[1]>h/4]

    for point in se:
        # for each point in each corner, find one with maximum number of neighbours as our target
        x,y = point[0],point[1]
        if direction == 1:
            num = len([i for i in se if x-300 <i[0]<x+50 and y-300 <i[1]<y+50])
        elif direction == 2:
            num = len([i for i in se if x-300 <i[0]<x+100 and y-50 <i[1]<y+300])
        elif direction == 3:
            num = len([i for i in se if x-50 <i[0]<x+300 and y-300 <i[1]<y+50])
        elif direction == 4:
            num = len([i for i in se if x-50 <i[0]<x+300 and y-50<i[1]<y+300])
        temp.append(num)
    idx = temp.index(max(temp))
    return se[idx]
        
def orthogonal_output(img, corners):
    # Calculate L2 norm for the length and height of the reoriented image
    width_ad = np.sqrt(((corners[0][0] - corners[3][0])**2) + ((corners[0][1] - corners[3][1])**2))
    width_bc = np.sqrt(((corners[1][0] - corners[2][0])**2) + ((corners[1][1] - corners[2][1])**2))
    max_width = max(int(width_ad), int(width_bc))
    height_ab = np.sqrt(((corners[0][0] - corners[1][0])**2) + ((corners[0][1] - corners[1][1])**2))
    height_cd = np.sqrt(((corners[2][0] - corners[3][0])**2) + ((corners[2][1] - corners[3][1])**2))
    max_height = max(int(height_ab), int(height_cd))
    
    # Specify the corners of the new image
    dstrect = np.array(
                [[0, 0],
                [0, max_height -1 ],
                [max_width - 1, max_height - 1],
                [max_width - 1, 0]], dtype = "float32")
    transform = cv2.getPerspectiveTransform(np.float32(corners), dstrect)
    warped_img = cv2.warpPerspective(img.copy(), transform, (max_width, max_height))
    return warped_img

# loop over all images
for image_name in all_images:
    try:  
        # print message
        print('Cropping image ' + image_name)
        
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

        # create bw image of detected green
        lower = np.array([36, 50, 30], dtype = "uint8")
        upper = np.array([86, 255, 255], dtype = "uint8")
        bw = cv2.inRange(hsv, lower, upper)
        
        # remove green from image to reduce noise
        img_no_green = np.copy(img)
        img_no_green[bw == 255] = 0

        # find bright pixel values that indicate frame and set to bw
        thres = cv2.threshold(cv2.cvtColor(img_no_green, cv2.COLOR_BGR2GRAY), brightness_thres_low, brightness_thres_high, cv2.THRESH_BINARY_INV)[1]
        thres = np.invert(thres)
        
        # adapt image to only contain frame
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        img_adapted = cv2.erode(thres,kernel, iterations = erode_it_1)
        img_adapted = cv2.dilate(img_adapted,kernel, iterations = dilate_it_1)
        img_adapted = cv2.erode(img_adapted, kernel, iterations = erode_it_2)

        h, w, _ = img.shape
        gray = cv2.GaussianBlur(img_adapted, (11, 11), 0)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200,)
        temp = img.copy()
        conv_line = []
        for i in range(len(lines)):
            for rho, theta in lines[i]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + w*(-b))
                y1 = int(y0 + w*(a))
                x2 = int(x0 - w*(-b))
                y2 = int(y0 - w*(a))
                cv2.line(temp,(x1,y1),(x2,y2),(0,255,255),5)
                a1 = y2-y1
                b1 = x1-x2
                c1 = a1*x1 +b1*y1
                conv_line.append([a1,b1,c1])
        valid_line = [[-(p2[2]*p1[1]-p1[2]*p2[1])/(p1[0]*p2[1]-p2[0]*p1[1]),-(p1[2]*p2[0]-p2[2]*p1[0])/(p1[0]*p2[1]-p2[0]*p1[1])] for p1,p2 in combinations(conv_line,2)
            if (p1[0]*p2[1]-p2[0]*p1[1]) != 0]
        
        valid_line = [i for i in valid_line if abs(i[1]) <= 6500 and abs(i[0]) <= 6500]
        
        # from filtered lines, search for the points in the intersection  
        valid_points = np.asarray([search_4points(valid_line, i, w, h) for i in range(1,5)])
        
        # Reorder points
        sums = [x[0] + x[1] for x in valid_points]
        top_left = valid_points[np.asarray(sums).argmin()]
        bottom_right = valid_points[np.asarray(sums).argmax()]

        low_y = np.argwhere(np.asarray([x[1] for x in valid_points]) < 750)
        top_right = valid_points[low_y[valid_points[low_y,0].argmax()]][0]

        low_x = np.argwhere(np.asarray([x[0] for x in valid_points]) < 750)
        bottom_left = valid_points[low_x[valid_points[low_x,1].argmax()]][0]
        
        corners = np.asarray([top_left, bottom_left, bottom_right, top_right])
        
        cropped_image = orthogonal_output(img, corners)
        cropped_image_clean = cropped_image[offset_top:(cropped_image.shape[0] - offset_bottom), offset_left:(cropped_image.shape[1] - offset_right)]
        
        if not os.path.exists(folder_path + 'cropped'):
            os.mkdir(folder_path + 'cropped')

        cv2.imwrite(folder_path + 'cropped' + '/' + image_name.replace('.JPG', '') + '_cropped.JPG', cropped_image_clean)

    except Exception as e:
        print("An error occurred:")
        print(e)
        print("Trying another method.")
        try: 
            # Add white border in case frame is outside of picture somewhere
            img_adapted[:,img.shape[1]-1] = 255
            img_adapted[:,0] = 255
            img_adapted[0,:] = 255
            img_adapted[img.shape[0]-1,:] = 255
            
            # find contours of frame
            img_cont = np.invert(img_adapted)
            img_cont = np.expand_dims(img_cont, axis = 2)
            contours, _ = cv2.findContours(img_cont, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            mask = np.zeros(img.shape)

            # find largest contoured part of image (assuming this is the area inside the frame)
            largestCon = 0
            i = 0
            area = 0
            for con in contours:
                new_area = cv2.contourArea(con)
                if(new_area > area):
                    area = new_area
                    largestCon = i
                i += 1
            
            # find corners of the the contour
            epsilon = 0.01 * cv2.arcLength(contours[largestCon], True)
            approximations = cv2.approxPolyDP(contours[largestCon], epsilon, True)
            
            # set corners of area inside frame
            c = approximations
            if(approximations.shape[0] > 4):
                sums = [sum(x[0], [1])[0] for x in c] 
                top_left = tuple(c[np.asarray(sums).argmin()][0])
                bottom_right = tuple(c[np.asarray(sums).argmax()][0])
                
                low_y = np.argwhere(np.asarray([x[0][1] for x in c]) < 750)
                top_right = tuple(c[low_y[c[low_y,:,0].argmax()]][0][0])
                
                low_x = np.argwhere(np.asarray([x[0][0] for x in c]) < 750)
                bottom_left = tuple(c[low_x[c[low_x,:,1].argmax()]][0][0])
            else:
                top_right = tuple(c[0][0])
                top_left = tuple(c[1][0])
                bottom_left = tuple(c[2][0])
                bottom_right = tuple(c[3][0])
            
            # crop image inside frame
            mask = np.ones(img.shape, dtype = np.uint8)
            mask = mask * 255
            corners = np.array([[top_left, bottom_left, bottom_right, top_right]], dtype = np.int32)
            cv2.fillPoly(mask, corners, 0)
            img_cropped = cv2.bitwise_or(img, mask)
            
            # Find image contour and find angle
            image_contour = contours[largestCon]
            minAreaRect = cv2.minAreaRect(image_contour)
            angle = minAreaRect[-1] - 90
            if angle < -45:
                angle = 90 + angle

            # Rotate image with angle
            img_rotated = img_cropped.copy()
            (h, w) = img_rotated.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_rotated = cv2.warpAffine(img_rotated, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            # Find contours of rotated image
            gray_rotated = img_rotated.copy()
            gray_rotated[:,img.shape[1]-1] = 255
            gray_rotated[:,0] = 255
            gray_rotated[0,:] = 255
            gray_rotated[img.shape[0]-1,:] = 255
            non_white = np.where((gray_rotated[:,:,0] != 255) | (gray_rotated[:,:,1] != 255) | (gray_rotated[:,:,2] != 255))
            gray_rotated[non_white] = [0, 0, 0]
            gray_rotated = cv2.cvtColor(gray_rotated, cv2.COLOR_BGR2GRAY)
            img_rotated_contours, _ = cv2.findContours(gray_rotated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            img_rotated_contours = sorted(contours, key = cv2.contourArea, reverse = True)[0]

            epsilon = 0.01 * cv2.arcLength(img_rotated_contours, True)
            corners = cv2.approxPolyDP(img_rotated_contours, epsilon, True)
            sums = [sum(x[0], [1])[0] for x in corners] 
            rotated_top_left = tuple(corners[np.asarray(sums).argmin()][0])
            rotated_bottom_right = tuple(corners[np.asarray(sums).argmax()][0])
            
            low_y = np.argwhere(np.asarray([x[0][1] for x in corners]) < 1000)
            rotated_top_right = tuple(corners[low_y[corners[low_y,:,0].argmax()]][0][0])
            
            low_x = np.argwhere(np.asarray([x[0][0] for x in corners]) < 1000)
            rotated_bottom_left = tuple(corners[low_x[corners[low_x,:,1].argmax()]][0][0])

            img_rotated_cropped = img_rotated[max(rotated_top_left[1] + offset_top, 0):(rotated_bottom_left[1] + offset_bottom), max(rotated_top_left[0] + offset_left, 0):(rotated_bottom_right[0] + offset_right)]

            if not os.path.exists(folder_path + 'cropped'):
                os.mkdir(folder_path + 'cropped')

            cv2.imwrite(folder_path + 'cropped' + '/' + image_name.replace('.JPG', '') + '_cropped.JPG', img_rotated_cropped)
            print("Image cropped, recommended to double check the output.")
            print("\n")
        except Exception as ex:
            print("An error occurred:")
            print(e)
            print("Cannot crop image " + image_name + ".")
