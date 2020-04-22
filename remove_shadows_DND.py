import cv2
import numpy as np 

def romove_shadows(image):
        
    b = image[:,:,0]
    g = image[:,:,1]
    r = image[:,:,2]
    rgb_planes = [b,g,r]
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_image = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_image = cv2.medianBlur(dilated_image, 21)
        diff_image = 255 - cv2.absdiff(plane, bg_image)
        norm_image = cv2.normalize(diff_image,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_image)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    normalised_image = cv2.merge(result_norm_planes)
    return normalised_image
    
