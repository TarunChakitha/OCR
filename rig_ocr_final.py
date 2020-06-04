import cv2
import numpy as np
import math
import pyttsx3
import pytesseract
import re
from deskew import determine_skew
from typing import Tuple, Union
from pytesseract import Output
from matplotlib import pyplot as plt
from PIL import Image, ImageMath
from blend_modes import divide
from spellchecker import SpellChecker

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

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

def get_median_angle(binary_image):
    erode_otsu = cv2.erode(binary_image,np.ones((7,7),np.uint8),iterations=1)
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
    #cv2.imwrite("negated_erode D.jpg",negated_erode)
    #cv2.imwrite("opening2 D.jpg",opening2)
    #cv2.imwrite("opening D.jpg",opening)
    #cv2.imwrite("double opening D.jpg",double_opening)
    #cv2.imwrite("double opening dialted 3x3 D.jpg",double_opening_dilated_3x3)
    #cv2.imwrite("sharp copy1.jpg",sharp_copy1)
    return median_angles


def complexAngle(angle):
        if 0 <= angle <= 90:
            corrected_angle = angle - 90
        elif -45 <= angle < 0:
            corrected_angle = angle - 90
        elif -90 <= angle < -45:
            corrected_angle = 90 + angle
        return corrected_angle

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

def get_otsu(image):
    '''
    _, otsu = cv2.threshold(image,180,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours_bin,_ = cv2.findContours(otsu,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    negated_otsu = ~otsu
    contours_bin_inv,_ = cv2.findContours(negated_otsu,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if (len(contours_bin) < len(contours_bin_inv)):
        print("white background image")
        return otsu
    else:
        print("black background image")
        return negated_otsu
    '''
    _, otsu = cv2.threshold(image,180,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu

def correct_skew(image):
    no_shadows = remove_shadows(image)
    image_resized = image_resize(no_shadows,2000,3000)
    
    gaussian_blur = cv2.GaussianBlur(image_resized,(5,5),0)

    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharp = cv2.filter2D(gaussian_blur,-1,kernel)
    #cv2.imwrite("no shadows.jpg",no_shadows)
    gray = cv2.cvtColor(sharp,cv2.COLOR_BGR2GRAY)
    otsu = get_otsu(gray)
    median_angles = get_median_angle(otsu)

    rotated_median_complex = rotate(image_resized,complexAngle(median_angles),(255,255,255))

    while True:
        rotated_median_complex_gray = cv2.cvtColor(rotated_median_complex,cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(rotated_median_complex_gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        osd_rotated_median_complex = pytesseract.image_to_osd(otsu)

        angle_rotated_median_complex = re.search('(?<=Rotate: )\d+', osd_rotated_median_complex).group(0)

        if (angle_rotated_median_complex == '0'):
            print("angle rotated median complex 0")
            print("no second rotation")
            rotated_median_complex = rotated_median_complex
            break
            #cv2.imshow("rotated median complex",rotated_median_complex)
            #cv2.imwrite("no second rotation.jpg",rotated_median_complex)
        elif (angle_rotated_median_complex == '90'):
            print("osd second rotation angle 90")
            rotated_median_complex = rotate(rotated_median_complex,90,(255,255,255))
            continue
            #cv2.imshow("second_rotation",second_rotation)
            #cv2.imwrite("second rotation.jpg",second_rotation)
        elif (angle_rotated_median_complex == '180'):
            print("osd second rotation angle 180")
            rotated_median_complex = rotate(rotated_median_complex,180,(255,255,255))
            continue
            #cv2.imshow("second_rotation",second_rotation)
            #cv2.imwrite("second rotation.jpg",second_rotation)
        elif (angle_rotated_median_complex == '270'):
            print("osd second rotation angle 270")
            rotated_median_complex = rotate(rotated_median_complex,90,(255,255,255))
            continue
            #cv2.imshow("second_rotation",second_rotation)
            #cv2.imwrite("second rotation.jpg",second_rotation)
    
    
    return rotated_median_complex

def de_shadow(image):
    bA = image[:,:,0]
    gA = image[:,:,1]
    rA = image[:,:,2]
    
    dilated_image_bB = cv2.dilate(bA, np.ones((7,7), np.uint8))
    dilated_image_gB = cv2.dilate(gA, np.ones((7,7), np.uint8))
    dilated_image_rB = cv2.dilate(rA, np.ones((7,7), np.uint8))
    bB = cv2.medianBlur(dilated_image_bB, 21)
    gB = cv2.medianBlur(dilated_image_gB, 21)
    rB = cv2.medianBlur(dilated_image_rB, 21)
    image = np.dstack((image, np.ones((image.shape[0],\
         image.shape[1], 1))*255))
    image = image.astype(float)
    dilate = [bB,gB,rB]
    dilate = cv2.merge(dilate)
    dilate = np.dstack((dilate, np.ones((image.shape[0],\
         image.shape[1], 1))*255))
    dilate = dilate.astype(float)
    
    blend = divide(image,dilate,1.0)
    blendb = blend[:,:,0]
    blendg = blend[:,:,1]
    blendr = blend[:,:,2]
    blend_planes = [blendb,blendg,blendr]
    blend = cv2.merge(blend_planes)
    blend = blend*0.85
    blend = np.uint8(blend)
    
    return blend

def list_to_string(list):
    str1 = " "
    return str1.join(list)

path_shadows = r'F:\tarun\images\shadows\shadows2.jpg'
path_skew = r'F:\tarun\images\skew\deskew-16.jpg'

image = cv2.imread(path_skew)
deskewed = correct_skew(image)
deskewed_copy = deskewed.copy()
image_resized = image_resize(deskewed,1600,1200)
image_resized_copy = image_resized.copy()
gray = cv2.cvtColor(image_resized,cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image_resized,cv2.COLOR_BGR2HSV)
v = hsv[:,:,2]
m = np.mean(v[:])
s = np.std(v[:])
k = -0.4
value = m + k*s

# niblack
val2 = m*(1+0.1*((s/128)-1))
print(value)

for p in range(image_resized.shape[0]):
    for q in range(image_resized.shape[1]):
        pixel = v[p,q]
        if (pixel > value):
            v[p,q] = 255
        else:
            v[p,q] = 0
v_copy = v.copy()

_,labels = cv2.connectedComponents(v)

result = np.zeros((v.shape[0],v.shape[1],3),np.uint8)

for i in range(labels.min(),labels.max()+1):
    mask = cv2.compare(labels,i,cv2.CMP_EQ)

    ctrs,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    result = cv2.drawContours(v_copy,ctrs,-1,(255,255,255))
cv2.imwrite("v copy.jpg",v_copy)


gray_copy = gray.copy()
ret, otsu = cv2.threshold(gray,180,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(ret)
coords_black = np.column_stack(np.where(gray < (ret + 10)))
coords_white = np.column_stack(np.where(gray > (ret + 10)))
print(len(coords_black),len(coords_white))

for i in range(len(coords_black)):
    gray_copy[coords_black[i][0],coords_black[i][1]] = 0
    
for i in range(len(coords_white)):
    gray_copy[coords_white[i][0],coords_white[i][1]] = 255


adaptive_gaussian = cv2.adaptiveThreshold(v,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,10)
custom_oem_psm_config = r'--oem 1 --psm 12'

ocr = pytesseract.image_to_data(v, output_type=Output.DICT,config=custom_oem_psm_config,lang='eng')
print(len(ocr['text']))

'''
    collecting text with confidence value > 64
                                               '''
def get_distance(x,y):
    return math.sqrt(math.pow(x,2) + math.pow(y,2))

xs = []
ys = []
ws = []
hs = []
texts = []
centers = []
confidences = []
bounding_boxes = []
distances = []
for i in range(len(ocr['text'])):
    if (int(ocr['conf'][i])>64):
        (x,y,w,h) = (ocr['left'][i],ocr['top'][i],ocr['width'][i],ocr['height'][i])
        xs.append(x)
        ys.append(y)
        ws.append(w)
        hs.append(h)
        #distances.append(get_distance(x,y))
        #confidences.append(ocr['conf'][i])
        #centers.append(w/2,h/2)
        texts.append(ocr['text'][i])
        bounding_box = x,y,w,h
        bounding_boxes.append([bounding_box])

ys,xs,ws,hs,texts = zip(*sorted(zip(ys,xs,ws,hs,texts)))

xs = list(xs)
ys = list(ys)
ws = list(ws)
hs = list(hs)
texts = list(texts)

# creating empty lists to store final(correctly sorted) bounding boxes
final_xs = []
final_ys = []
final_ws = []
final_hs = []
final_texts = []
indices = []

xs_ini = xs[0]
ys_ini = ys[0]
ws_ini = ws[0]
hs_ini = hs[0]

mean_hs = np.mean(hs)

count = 0
flag = 0
#while True:
for j in range(len(ys)):
    #print(len(ys))
    if j>0:
        for a in range(len(indices)):
            if j == a:
                print(j," == ",a)
                flag = 1
                break
    if flag == 1:
        flag = 0
        print("continued")
        continue
    print("should not print if continued")
    count = count + 1

    # initialising the top bounding box of top line to get all first line bounding boxes
    xs_ini = xs[j]
    ys_ini = ys[j]
    ws_ini = ws[j]
    hs_ini = hs[j]
    #print(xs_ini,ys_ini,ws_ini,hs_ini)

    # creating empty temporary lists to store all bounding boxes of 1st line
    temp_xs = []
    temp_ys = []
    temp_ws = []
    temp_hs = []
    temp_texts = []
    indices = []
    # getting the 1st line bounding boxes using the given condition
    for i in range(len(ys)):
        if (abs(ys[i] - ys_ini) < (mean_hs)):
            print("appended 1",i)
            temp_ys.append(ys[i])
            temp_xs.append(xs[i])
            temp_ws.append(ws[i])
            temp_hs.append(hs[i])    
            temp_texts.append(texts[i])
            indices.append(i)
            #ys.remove(ys[i])
            #hs.remove(hs[i])
        # elif (((ys_ini + hs_ini) > (ys[i] + hs[i])) and ((ys[i] + hs[i])  > ys_ini)):
        #     print("appended 1",i)
        #     temp_ys.append(ys[i])
        #     temp_xs.append(xs[i])
        #     temp_ws.append(ws[i])
        #     temp_hs.append(hs[i])    
        #     temp_texts.append(texts[i])
        #     indices.append(i)
        #     #ys.remove(ys[i])
        #     #hs.remove(hs[i])

    # sorting the 1st line bounding boxes according to x coordinate (left-to-right)
    #if (len(temp_xs) != 0 or len(temp_ys) != 0 or len(temp_ws) != 0 or len(temp_hs) != 0 or len(temp_texts) != 0):
    temp_xs,temp_ys,temp_ws,temp_hs,temp_texts = zip(*sorted(zip(temp_xs,temp_ys,temp_ws,temp_hs,temp_texts)))
    final_xs = final_xs + list(temp_xs)
    final_ys = final_ys + list(temp_ys)
    final_ws = final_ws + list(temp_ws)
    final_hs = final_hs + list(temp_hs)
    final_texts = final_texts + list(temp_texts)
    print("Indices: ", len(indices))
    # if(len(indices) != 0):
    #     for k in indices:
    #         xs.pop(k)
    #         # print("k = " , k)
    #         ys.pop(k)
    #         ws.pop(k)
    #         hs.pop(k)
    #         texts.pop(k)

    # if (len(ys) == 0):
    #     # global count
    #     print(count)
    #     return 0
    # else:


#spell = SpellChecker()

boxes = len(final_texts)
engine = pyttsx3.init()
for i in range(boxes):
    #ocr['text'][i] = spell.correction(ocr['text'][i])
    #x,y,w,h = xs[i],ys[i],ws[i],hs[i]
    x,y,w,h = final_xs[i],final_ys[i],final_ws[i],final_hs[i]
    cv2.rectangle(image_resized_copy,(x,y),(x+w,y+h),(0,0,255),1)
    cv2.imshow("text",image_resized_copy)
    engine.say(final_texts[i])
    engine.runAndWait()
    #cv2.waitKey(500)

string = list_to_string(final_texts)
print("String: ",string)
div_mer_img = de_shadow(deskewed_copy)
cv2.imwrite("div mer.jpg",div_mer_img)
cv2.imwrite("gray.jpg",gray)
cv2.imwrite("gray copy.jpg",gray_copy)
cv2.imwrite("adaptive gaussian.jpg",adaptive_gaussian)
cv2.imwrite("v.jpg",v)
cv2.imwrite("text.jpg",image_resized_copy)