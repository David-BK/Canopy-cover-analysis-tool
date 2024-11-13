# Canopy Cover Analysis Tool

v0.6
doi: 10.5281/zenodo.13941037

This tool is meant to analyse plant canopy cover images through image cropping and proportion green detection.

This tool uses Python (recommended version: 3.11) and the packages numpy, matplotlib, opencv-python, pillow, scikit-image, and pillow_heif.

## Canopy cover analysis
To combine both tools (image cropping and green detection, see below), use canopy_cover.py:  
1: Open CMD or PowerShell  
2: Navigate to canopy_cover.py  
3: Run as follows:  
python canopy_cover.py -f <path_to_folder_with_images>  

Or  

python canopy_cover.py --folderpath <path_to_folder_with_images>  

Example:  
python canopy_cover.py -f "C:/Projects/My_Canopy_Project/Canopy_Cover_Images"  

To print the help:  
python canopy_cover.py -h  

Or  

python canopy_cover.py --help  

Additional (optional) arguments can be given as described for the individual tools below.  
  
## Image cropping

### How to use the tool  
1: Open CMD or PowerShell  
2: Navigate to crop_frame_image.py  
3: Run as follows:  
python crop_frame_image.py -f <path_to_folder_with_images>  

Or  

python crop_frame_image.py --folderpath <path_to_folder_with_images>  

Example:  
python crop_frame_image.py -f "C:/Projects/My_Canopy_Project/Canopy_Cover_Images"  

To print the help:  
python crop_frame_image.py -h  

Or  

python crop_frame_image.py --help  
  
#### Additional optional arguments (read the script on how these parameters are used specifically):  
-k --kernelsize: An integer. Size of the kernel used for erosion and dilation modifications to detect the frame. Default is the rounded integer of image width divided by 4000 (10 for an image of size 3000x4000).  
  
-w --colourthreslow: An integer. Lower threshold of the colour detection for the frame. Default is 130.  
  
-i --colourthreshigh: An integer. Upper threshold of the colour detection for the frame. Default is 255.  
  
-r --erode1: An integer. Number of iterations of the first erosion operation on the image to detect the frame. Default is 4.  
  
-d --dilate: An integer. Number of iterations of the dilation operation on the image to detect the frame. Default is 4.  
  
-u --erode2: An integer. Number of iterations of the second erosion operation on the image to detect the frame. Default is 1.  
  
-t --offtop: An integer. Number of pixels to be offset at the top of the picture (in case a part of the frame is visible in every cropped image). Default is the rounded integer of image width divided by 60 (67 for an image of size 3000x4000).  
  
-b --offbot: An Integer. Number of pixels to be offset at the bottom of the picture. Default is the rounded integer of image width divided by 60 (67 for an image of size 3000x4000).  
  
-x --offleft: An Integer. Number of pixels to be offset at the left of the picture. Default is the rounded integer of image width divided by 60 (67 for an image of size 3000x4000).  
  
-y --offright: An Integer. Number of pixels to be offset at the right of the picture.Default is the rounded integer of image width divided by 60 (67 for an image of size 3000x4000).  

## Green detection

### How to use the tool  
1: Open CMD or PowerShell  
2: Navigate to green_detection.py  
3: Run as follows:  
python green_detection.py -f <path_to_folder_with_cropped_images>  

Or  

python green_detection.py --folderpath <path_to_folder_with_cropped_images>  

### Optional arguments:
-o --output: A string. The location and name of the output file. Default is <folderpath>/canopycover.txt.  

-c --control: A boolean. Whether to output control black and white images into <folderpath>/bw/ folder. Default is False.  

Example:  
python green_detection.py -f "C:/Projects/My_Canopy_Project/Canopy_Cover_Images/cropped" -o "C:/Projects/My_Canopy_Project/canopy_cover.txt" -c True

To print the help:  
python green_detection.py -h  

Or  

python green_detection.py --help  

#### Additional optional arguments (read the script on how these parameters are used specifically):  
-u --huelower: An integer. Lower limit of hue in HSV image for green detection. Default is 36.  
  
-e --hueupper: An integer. Upper limit of hue in HSV image for green detection. Default is 86.  
  
-s --satlower: An integer. Lower limit of saturation in HSV image for green detection. Default is 50.  
  
-a --satupper: An integer. Upper limit of saturation in HSV image for green detection. Default is 255.  
  
-v --vallower: An integer. Lower limit of value in HSV image for green detection. Default is 30.  
  
-l --valupper: An integer. Lower limit of value in HSV image for green detection. Default is 255.  

-p --detectwhite: A boolean. Whether or not to include the detection of bright white colours, e.g. when to include white flowers. Default is False.

-n --noisereduction: An integer. How much noise reduction to apply, i.e. the maximum area of detected objects to remove. Default is 10.

#### Additional questions  
If you have any additional questions, need help, or have suggestions for improvements, please contact me: david.kottelenberg@wur.nl  

## Click cropping
Alternatively to the automatic cropping, you can use the click cropping tool to manually crop the images in a quick and easy way by clicking the corners of the frame.  

### How to use the tool  
1: Open CMD or PowerShell  
2: Navigate to click_crop.py  
3: Run as follows:  
python click_crop.py -f <path_to_folder_with_cropped_images>  

Or  

python click_crop.py --folderpath <path_to_folder_with_cropped_images>  

By running the file, the images will open one by one. After clicking the corners of an image, close the image and the next one will be opened. The cropped images will be placed in the 'cropped' folder in the image directory.  

Example:  
python click_crop.py -f "C:/Projects/My_Canopy_Project/Canopy_Cover_Images/"  
