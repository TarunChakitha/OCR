import numpy as np 
import cv2 
from deskew import determine_skew, determine_skew_dev
from typing import Tuple, Union
import math

def rotate(image: np.ndarray, background_color):
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
    
class RRect:
  def __init__(self, p0, s, ang):
    self.p0 = (int(p0[0]),int(p0[1]))
    (self.W, self.H) = s
    self.ang = ang
    self.p1,self.p2,self.p3 = self.get_verts(p0,s[0],s[1],ang)
    self.verts = [self.p0,self.p1,self.p2,self.p3]

  def get_verts(self, p0, W, H, ang):
    sin = np.sin(ang/180*3.14159)
    cos = np.cos(ang/180*3.14159)
    P1 = (int(self.H*sin)+p0[0],int(self.H*cos)+p0[1])
    P2 = (int(self.W*cos)+P1[0],int(-self.W*sin)+P1[1])
    P3 = (int(self.W*cos)+p0[0],int(-self.W*sin)+p0[1])
    return [P1,P2,P3]

  def draw(self, image):
    print(self.verts)
    for i in range(len(self.verts)-1):
      cv2.line(image, (self.verts[i][0], self.verts[i][1]), (self.verts[i+1][0],self.verts[i+1][1]), (0,255,0), 2)
    cv2.line(image, (self.verts[3][0], self.verts[3][1]), (self.verts[0][0], self.verts[0][1]), (0,255,0), 2)

