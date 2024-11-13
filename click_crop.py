# import packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import fnmatch
import sys
import getopt
from itertools import combinations
import math
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()

def click_event(event, x, y, flags, params):
  
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
  
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
  
        cv2.imshow('image', img)
        corners.append([x, y])
        return corners
        
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
        
# driver function
if __name__=="__main__":
    # setup input and parameters
    folder_path = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hf:", ["help", "folderpath"])
    except getopt.GetoptError:
        print('An unkown error occurred. Default usage: ')
        print('click_crop.py -f <folderpath>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('Usage: \nclick_crop.py -f <folderpath>')
            print('\n')
            sys.exit()
        elif opt in ("-f", "--folderpath"):
            folder_path = arg
            folder_path = folder_path.replace(os.sep, '/')
        
    if not folder_path[-1] == '/':
        folder_path = folder_path + '/'
        
    all_images = []
    for file in os.listdir(folder_path):
        if fnmatch.fnmatch(file, '*.JPG') or fnmatch.fnmatch(file, '*.HEIC') or fnmatch.fnmatch(file, '*.PNG'):
            all_images.append(file)

    for image_name in all_images:
        try: 
        
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

            # Loads an image
            cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
            # displaying the image
            cv2.imshow('image', img)
            cv2.resizeWindow('image', 600, 800)
          
            # setting mouse handler for the image
            # and calling the click_event() function
            corners = []
            cv2.setMouseCallback('image', click_event)
            # wait for a key to be pressed to exit
            cv2.waitKey(0)
            corners = np.asarray(corners)
          
            # Sort the top two points based on x-coordinates (left to right)
            top_points = sorted(corners[:2], key=lambda p: p[0])
            # Sort the bottom two points based on x-coordinates (left to right)
            bottom_points = sorted(corners[2:], key=lambda p: p[0])
            # Reassign points in the correct order
            top_left, top_right = top_points
            bottom_left, bottom_right = bottom_points
            
            corners_ordered = np.asarray([top_left, bottom_left, bottom_right, top_right])
            cropped_image = orthogonal_output(img, corners_ordered)
            
            if not os.path.exists(folder_path + 'cropped'):
                os.mkdir(folder_path + 'cropped')

            cv2.imwrite(folder_path + 'cropped' + '/' + image_name.replace('.JPG', '') + '_cropped.JPG', cropped_image)
                
            # close the window
            cv2.destroyAllWindows()
        except Exception as e:
            print("An error occurred:")
            print(e)
    
