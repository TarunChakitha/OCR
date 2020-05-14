import cv2
import numpy as np
import math
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
for f in range(0,30):
    name = ""
    print(name)
    image = cv2.imread("bw.jpg")
    image = cv2.resize(image, (800,600), interpolation = cv2.INTER_AREA)
    # cv2.imshow("original",image)
    ima = image.copy()
    ima2 = image.copy()
    ima3 = image.copy()
    rgb_planes = cv2.split(image)
    result_planes_blur = []
    result_planes_dilate = []
    result_norm_planes_blur = []
    result_norm_planes_dilate = []
    for plane in rgb_planes :
        # For removing shadows
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        # cv2.imshow('dila', dilated_img)
        # cv2.waitKey(0)
        blur_img = cv2.medianBlur(plane, 21)
        # cv2.imshow('blur', blur_img)
        blur_img_2 = cv2.medianBlur(blur_img, 21)
        # cv2.imshow('blur2', blur_img_2)
        blur_dil_img = cv2.medianBlur(dilated_img,21)
        # cv2.imshow("blur_dil_img",blur_dil_img)
        #cv2.waitKey(0)
        diff_img_blur = 255 - cv2.absdiff(plane, blur_img_2)
        diff_img_dilate = 255-cv2.absdiff(plane,blur_dil_img)
        # cv2.imshow("diff_blur2",diff_img_blur)
        # cv2.imshow("diff_dil",diff_img_dilate)
        #cv2.waitKey(0)
        norm_img_blur = np.zeros((diff_img_blur.shape[0],diff_img_blur.shape[1], 1), dtype = np.uint8)
        norm_img_dilate = np.zeros((diff_img_dilate.shape[0],diff_img_dilate.shape[1], 1), dtype = np.uint8)
        cv2.normalize(diff_img_blur, norm_img_blur , 0, 255,cv2.NORM_MINMAX,dtype = cv2.CV_8UC1)
        cv2.normalize(diff_img_dilate, norm_img_dilate, 0, 255,cv2.NORM_MINMAX,dtype = cv2.CV_8UC1)
        # cv2.imshow("norm_img_blur",norm_img_blur)
        # cv2.imshow("norm_img_dilate",norm_img_dilate)
        #cv2.waitKey(0)
        result_planes_blur.append(diff_img_blur)
        result_norm_planes_blur.append(norm_img_blur)
        result_planes_dilate.append(diff_img_dilate)
        result_norm_planes_dilate.append(norm_img_dilate)

    result_blur = cv2.merge(result_planes_blur)
    result_norm_blur = cv2.merge(result_norm_planes_blur)
    result_dilate = cv2.merge(result_planes_dilate)
    result_norm_dilate = cv2.merge(result_norm_planes_dilate)
    # cv2.imshow('norm_blur', result_norm_blur) #final image but with the outline of the finger
    # cv2.imshow('norm_dilate', result_norm_dilate)
    # cv2.imshow('diff_blur', result_blur)
    # cv2.imshow('diff_dilate', result_dilate)
    #cv2.waitKey(0)


    '''
    Exponential Transform
                            '''
    gray_blur = cv2.cvtColor(result_norm_blur,cv2.COLOR_BGR2GRAY)
    gray_dil = cv2.cvtColor(result_norm_dilate,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray_blur',gray_blur)
    # cv2.imshow('gray_dilate',gray_dil)
    #cv2.waitKey(0)
    processed_image_blur = gray_blur*0.02
    # cv2.imshow("multi",processed_image_blur)
    #cv2.waitKey(0)
    out_blur = np.exp(processed_image_blur)
    # cv2.imshow("expo_blur",out_blur)
    #cv2.waitKey(0)
    out_blur = out_blur - 1
    # cv2.imshow("sub_blur",out_blur)
    #cv2.waitKey(0)
    cv2.normalize(out_blur, out_blur, 0, 255, cv2.NORM_MINMAX)
    # cv2.imshow("norm_blur",out_blur)
    #cv2.waitKey(0)
    inv_blur = 255 - out_blur
    inv_blur = np.uint8(inv_blur)
    # cv2.imshow("inverted_blur",inv_blur)
    #cv2.waitKey(0)
    ret,thresh_blur = cv2.threshold(inv_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imshow("afterthresh_otsu_blur",thresh_blur)
    # cv2.waitKey(0)
    processed_image_dil = gray_dil
    processed_image_dil = processed_image_dil*0.02
    # cv2.imshow("multi_dil",processed_image_dil)
    #cv2.waitKey(0)
    out_dil = np.exp(processed_image_dil)
    # cv2.imshow("expo_dil",out_dil)
    #cv2.waitKey(0)
    out_dil = out_dil - 1
    # cv2.imshow("sub_dil",out_dil)
    #cv2.waitKey(0)
    cv2.normalize(out_dil, out_dil, 0, 255, cv2.NORM_MINMAX)
    # cv2.imshow("norm_dil",out_dil)
    #cv2.waitKey(0)
    inv_dil = 255 - out_dil
    inv_dil = np.uint8(inv_dil)
    # cv2.imshow("inverted_dil",inv_dil)
    #cv2.waitKey(0)
    ret4,thresh_dil=cv2.threshold(inv_dil,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imshow("afterthresh_otsu_dil",thresh_dil)
    # cv2.waitKey(0)
    res =cv2.bitwise_and(thresh_blur,thresh_dil)
    cv2.imshow("Res",res)
    cv2.imwrite("res1.jpeg",res)
    if cv2.waitKey(0) == 27:
        break
