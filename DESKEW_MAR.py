import numpy as np 
import cv2 
import math
import pyttsx3
import pytesseract
import IMAGE_TOOLS_LIB
import re
from deskew import determine_skew
from typing import Tuple, Union
from pytesseract import Output
from matplotlib import pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
path_shadows = r'F:\tarun\images\shadows\shadows1.jpg'
path_skew = r'F:\tarun\images\skew\deskew-25.jpg'

def correct_skew(image):
    #image = cv2.imread("2.jpeg")
    image_resized = IMAGE_TOOLS_LIB.image_resize(image,2000,3000)
    image_resized1 = image_resized.copy()
    no_shadows = IMAGE_TOOLS_LIB.remove_shadows(image_resized)
    gaussian_blur = cv2.GaussianBlur(no_shadows,(5,5),0)

    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    #kernel_excessive = np.array([[1,1,1],[1,-7,1],[1,1,1]])
    sharp = cv2.filter2D(gaussian_blur,-1,kernel)
    gray = cv2.cvtColor(sharp,cv2.COLOR_BGR2GRAY)
    #gray_copy = gray.copy()
    #sharp_copy = sharp.copy()
    #sharp_copy2 = sharp.copy()
    #adaptive = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,55,25)
    def get_otsu(image):
        ret, otsu = cv2.threshold(gray,180,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours_bin,_ = cv2.findContours(otsu,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        negated_otsu = ~otsu
        contours_bin_inv = cv2.findContours(negated_otsu,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        if (len(contours_bin) > len(contours_bin_inv)):
            print("white backgund image")
            return otsu
        else:
            print("black backgund image")
            return negated_otsu

    otsu = get_otsu(gray)
    #skew = determine_skew(gray)

    erode_otsu = cv2.erode(otsu,np.ones((7,7),np.uint8),iterations=1)
    negated_erode = ~erode_otsu
    opening = cv2.morphologyEx(negated_erode,cv2.MORPH_OPEN,np.ones((5,5),np.uint8),iterations=2)
    double_opening = cv2.morphologyEx(opening,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations=5)
    double_opening_dilated_3x3 = cv2.dilate(double_opening,np.ones((3,3),np.uint8),iterations=4)
    contours_otsu,hierarchy = cv2.findContours(double_opening_dilated_3x3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    angles = []

    for cnt in range(len(contours_otsu)):
        rect = cv2.minAreaRect(contours_otsu[cnt])
        angles.append(rect[-1])
        
    angles.sort()
    median_angles = np.median(angles)

    #print("median of angles = ",median_angles)

    def complexAngle(angle):
        if 0 <= angle <= 90:
            corrected_angle = angle - 90
        elif -45 <= angle < 0:
            corrected_angle = angle - 90
        elif -90 <= angle < -45:
            corrected_angle = 90 + angle
        return corrected_angle

    #print("corrected deskew library",skew)
    #print("corrected mar median angle complex", complexAngle(median_angles))

    def rotate(image: np.ndarray,angle, background_color): # OFFIAL DOCUMENTATION
        old_width, old_height = image.shape[:2]
        angle_radian = math.radians(angle)
        width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
        height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)  
        rot_mat[1, 2] += (width - old_width) / 2
        rot_mat[0, 2] += (height - old_height) / 2
        return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background_color)


    #rotated_deskew = rotate(image_resized,skew,(0,0,0))
    rotated_median_complex = rotate(image_resized,complexAngle(median_angles),(255,255,255))

    rotated_median_complex_gray = cv2.cvtColor(rotated_median_complex,cv2.COLOR_BGR2GRAY)
    rotated_median_complex_gaussian = cv2.adaptiveThreshold(rotated_median_complex_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,55,25)

    osd_rotated_median_complex = pytesseract.image_to_osd(rotated_median_complex_gaussian)
    angle_rotated_median_complex = re.search('(?<=Rotate: )\d+', osd_rotated_median_complex).group(0)
    #print("angle: rotated median complex", angle_rotated_median_complex)

    if (angle_rotated_median_complex == '0'):
        print("angle rotated median complex 0")
        print("no second rotation")
        second_rotation = rotated_median_complex
        #cv2.imshow("rotated median complex",rotated_median_complex)
        #cv2.imwrite("no second rotation.jpg",rotated_median_complex)
    elif (angle_rotated_median_complex == '90'):
        print("osd second rotation angle 90")
        second_rotation = rotate(rotated_median_complex,90,(255,255,255))
        #cv2.imshow("second_rotation",second_rotation)
        #cv2.imwrite("second rotation.jpg",second_rotation)
    elif (angle_rotated_median_complex == '180'):
        print("osd second rotation angle 180")
        second_rotation = rotate(rotated_median_complex,180,(255,255,255))
        #cv2.imshow("second_rotation",second_rotation)
        #cv2.imwrite("second rotation.jpg",second_rotation)
    elif (angle_rotated_median_complex == '270'):
        print("osd second rotation angle 270")
        second_rotation = rotate(rotated_median_complex,90,(255,255,255))
        #cv2.imshow("second_rotation",second_rotation)
        #cv2.imwrite("second rotation.jpg",second_rotation)
    
    return second_rotation