import numpy as np 
import cv2 
from deskew import determine_skew, determine_skew_dev
from typing import Tuple, Union
import math
import pyttsx3
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
path_shadows = r'F:\tarun\images\shadows\shadows1.jpg'
path_skew = r'F:\tarun\images\skew\deskew-9.jpg'


def image_resize(image: np.ndarray, width = None, height = None, inter = cv2.INTER_CUBIC):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def remove_shadows(image: np.ndarray):
    b = image[:,:,0]
    g = image[:,:,1]
    r = image[:,:,2]
    rgb_planes = [b,g,r]
    #result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_image = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_image = cv2.medianBlur(dilated_image, 21)
        diff_image = 255 - cv2.absdiff(plane, bg_image)
        norm_image = cv2.normalize(diff_image,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        #result_planes.append(diff_image)
        result_norm_planes.append(norm_image)

    #result = cv2.merge(result_planes)
    normalised_image = cv2.merge(result_norm_planes)
    return normalised_image

def rotate(image: np.ndarray, background_color): # OFFIAL DOCUMENTATION
    old_width, old_height = image.shape[:2]
    gray_scale = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    angle = determine_skew(gray_scale) 
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)
    #print(old_width,old_height)
    #print(width,height)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)  
    #print(rot_mat[1, 2],rot_mat[0, 2])
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    #print(rot_mat[1, 2],rot_mat[0, 2])
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background_color),angle


image = cv2.imread(path_skew)
image_resized = image_resize(image,1200,675)
no_shadows = remove_shadows(image_resized)
deskewed ,angle = rotate(image_resized,(255,255,255))
gaussian_blur = cv2.GaussianBlur(no_shadows,(5,5),0)
#gaussian_blur = cv2.GaussianBlur(deskewed,(7,7),0)

kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
kernel_excessive = np.array([[1,1,1],[1,-7,1],[1,1,1]])
sharp = cv2.filter2D(gaussian_blur,-1,kernel)
gray = cv2.cvtColor(sharp,cv2.COLOR_BGR2GRAY)
sharp_copy = sharp.copy()

#clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
#equalised = clahe.apply(gray)

#gaussian = cv2.GaussianBlur(deskewed,(5,5),0)
#smooth = cv2.addWeighted(gaussian,1.5,deskewed,-0.5,0)
#bilateral = cv2.bilateralFilter(deskewed,9,75,75)

binary_gaussian = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,55,25)
binary_mean = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,55,25)
ret, otsu = cv2.threshold(gray,180,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#print(ret)

#dilate = cv2.dilate(binary_gaussian,np.ones((3,3),np.uint8))
erode = cv2.erode(binary_gaussian,np.ones((5,5),np.uint8))
#closing = cv2.morphologyEx(binary_gaussian,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))

custom_oem_psm_config1 = r'--oem 3 --psm 12'
#custom_oem_psm_config2 = r'--oem 1 --psm 12'
#custom_oem_psm_config2 = r'--oem 3 --psm 11'

#ocr1 = pytesseract.image_to_data(gray1, output_type=Output.DICT,config=custom_oem_psm_config1,lang='eng')
ocr = pytesseract.image_to_data(gray, output_type=Output.DICT,config=custom_oem_psm_config1,lang='eng')
#ocr3= pytesseract.image_to_data(gray3, output_type=Output.DICT,config=custom_oem_psm_config2,lang='eng')
#print(d.keys())
print(len(ocr['text']))

boxes = len(ocr['text'])
#engine = pyttsx3.init()
for i in range(boxes):
    if int(ocr['conf'][i])>60:
        #print(d2['text'][i])
        (x,y,w,h) = (ocr['left'][i],ocr['top'][i],ocr['width'][i],ocr['height'][i])
        cv2.rectangle(sharp_copy,(x,y),(x+w,y+h),(0,0,255),1)
        #cv2.imshow("text",erode)
        #engine.say(d2['text'][i])
        #engine.runAndWait()
        #cv2.waitKey(10)
'''
h, w = gray.shape
boxes = pytesseract.image_to_boxes(gray) 
for b in boxes.splitlines():
    b = b.split(' ')
    cv2.rectangle(sharp_copy, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)


'''


cv2.imshow("resized",image_resized)
cv2.imshow("deskewed",deskewed)
cv2.imshow("text",sharp_copy)
#cv2.imshow("gaussia",gaussian)
#cv2.imshow("equalised",equalised)
#cv2.imshow("gray",gray2)
cv2.imshow("sharp",sharp)
#cv2.imshow("binary_mean",binary_mean)
#cv2.imshow("binary_gaussian",binary_gaussian)
#cv2.imshow("otsu",otsu)
#cv2.imshow("dilate",dilate)
#cv2.imshow("erode",~erode)
#cv2.imshow("hist_eq",cl1)
print(angle)
cv2.waitKey(0)
cv2.destroyAllWindows()


