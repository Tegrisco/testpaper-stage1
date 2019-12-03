"""
=====================================
Get the coordinates of kousuanti
=====================================

A script for Python3.

Use Tesseract OCR and template matching to locate the kousuanti region.
结合Tesseract OCR 和模版匹配定位口算题区域。

"""

# Python Imaging Library
from PIL import Image

# Numeric Python
import numpy as np

# opencv
import cv2

# regular expression 
import re

# Switch locale require
import locale
from contextlib import contextmanager


@contextmanager
def c_locale():
    """Switch locale.

    Temporarily change the operating locale to accommodate Tesseract.
    This locale is automatically finalized when used in a with-statement
    (context manager).

    Args: None
    Returns: None
    Raises: None

    """
    #print ('Enter c_locale')
    try:
        currlocale = locale.getlocale()
    except ValueError:
        #print('Manually set to en_US UTF-8')
        currlocale = ('en_US', 'UTF-8')
    #print ('Switching to C from {}'.format(currlocale))
    locale.setlocale(locale.LC_ALL, "C")
    yield
    #print ('Switching to {} from C'.format(currlocale))
    locale.setlocale(locale.LC_ALL, currlocale)


def get_template(imagePath, p):
    """Fetches symbol '=' template coordinates from image by Tesseract OCR.

    Recognize the rectangle of kousuanti by TEXTLINE at first, then recognize 
    line which contains '=' by SYMBOL. Therefore the width of template is '='
    symbol's width and the height of template is line's height.

    Args:
        imagePath: The file path of image.
        p: The position of kousuanti.

    Returns:
        An array with four integer elements includes the coordinates of 
        the upper left corner and the lower right corner of the template 
        region. For example:

        [1, 2, 3, 4]

    Raises: 
        IOError: An error occurred accessing the imagePath object.
        ValueError: An error occurred getting the value of p object.

    """
    img = cv2.imread(imagePath)
    imgPil = Image.open(imagePath)
    cropImage = []
    position = []
    #cropImage = imgPil.crop((p[0],p[1],p[2],p[3]))
    #cropImage.show()
    with c_locale():
        from tesserocr import PyTessBaseAPI, RIL, iterate_level, PSM
        with PyTessBaseAPI(psm=PSM.SINGLE_BLOCK) as api:
            api.SetImageFile(imagePath)
            api.SetRectangle(p[0], p[1], p[2]-p[0], p[3]-p[1])
            api.Recognize()
            ri = api.GetIterator()
            level = RIL.TEXTLINE
            #text = api.GetUTF8Text()
            #print (text)
            for r in iterate_level(ri, level):
                text = r.GetUTF8Text(level)
                conf = r.Confidence(level)
                box = r.BoundingBox(level)
                if re.search(u'=',text) :
                    cropImage = imgPil.crop((box[0],box[1],box[2],box[3]))
                    #cropImage.show()
                    #cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(255,0,0),1)
                    with PyTessBaseAPI(psm=PSM.SINGLE_LINE) as api1:
                        api1.SetImageFile(imagePath)
                        api1.SetRectangle(box[0], box[1], box[2] - box[0], box[3] - box[1])
                        api1.Recognize()
                        ri1 = api1.GetIterator()
                        level1 = RIL.SYMBOL
                        for r1 in iterate_level(ri1, level1):
                            text1 = r1.GetUTF8Text(level1)
                            conf1 = r1.Confidence(level1)
                            box1 = r1.BoundingBox(level1)
                            #print ('box1 = {}'.format(box1))
                            #print ('text1:\n{}'.format(text1))
                            if text1 == u'=' :
                                #print ('box1:\n{}'.format(box1))
                                position = [box1[0], box[1], box1[2], box[3]]
                                # cropImage = imgPil.crop(tuple(position))
                                #cropImage.show()
                                break
                    break
    return position


def preprocess_bw_inv(gray, smooth=True):
    """Morphological operation.

    Args: 
        gray: gray image.
        smooth: smooth image or not

    Returns:
        An image.

    Raises: 
        ValueError: An error occurred getting the value of gray.

    """
    #print ('Enter c_locale')
    if smooth:
        gray = cv2.boxFilter(gray, -1, (3, 3))

    ret, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw2 = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, element)

    return bw2


def vertical_proj(bw):
    """Fetches vertical weight.

    Args: 
        bw: Binary image.

    Returns:
        An array of vertical weight of image.

    Raises: 
        ValueError: An error occurred getting the value of bw.

    """
    # 一个像素点权重算1，不算255
    bw = bw//255
    # weight 是每一列的重量
    weight = np.sum(bw, axis=0)
    return weight


def template_match(imagePil, template, threshold=0.8):
    """Match template.

    Args: 
        imagePil: PIL image.
        template: Template image.
        threshold: Matching threshold.

    Returns:
        An dic of coordinate array with four integer elements includes 
        the coordinates of the upper left corner and the lower right corner. 
        For example:

        {1:[1,2,3,4], 2:[1,2,3,4]}

    Raises: 
        ValueError: An error occurred getting the value of imagePil and template.

    """
    img_rgb = np.array(imagePil.convert('RGB'))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = np.array(template.convert('RGB'))
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    #image_show('template',template_gray)
    coordinate = {}
    num = 0
    w, h = template_gray.shape[::-1]
    #h, w = template.shape[:2]  # rows->h, cols->w
    res = cv2.matchTemplate(img_gray,template_gray,cv2.TM_CCOEFF_NORMED)
    #threshold = 0.85
    #print ('threshold = {}'.format(threshold))
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        coordinate[num] = [int(pt[0]), int(pt[1]), int(pt[0] + w), int(pt[1] + h)]
        num += 1
        #cv2.rectangle(img_rgb, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), 0, 1)
    num = 0
    coorSort = sortPoint(coordinate)
    for i,j in coorSort.copy().items():
        #previous = i - 1
        next = i + 1
        diff = abs(coorSort[next][0]-j[0]) if next in coorSort else 201
        if diff > 200:
            coorSort[num] = coorSort[i]
            num += 1
        del coorSort[i]
    return coorSort


def setSortRule(p1, p2):
    """Determine whether two coordinates are in the same row.

    p1 and p2 is in the same row if the difference between their ordinate 
    of upper left corner is less than 20pixel.

    Args: 
        p1: A coordinate.
        p2: Another coordinate.

    Returns:
        Boolean variable.

    Raises: None

    """
    if abs(p1[1] - p2[1]) <= 20:
        return (p1[0] > p2[0])
    else:
        return False


def sortPoint(dic):
    """Sort coordinates from top to bottom.

    Args: 
        dic: A dic of coordinates.

    Returns: 
        Sorted dic.

    Raises: None

    """
    length = len(dic)
    for i in range(0, length-1):
        #import pdb 
        #pdb.set_trace()
        for j in range(0, length-1-i):
            if setSortRule(dic[j], dic[j+1]):
                dic[j], dic[j+1] = dic[j+1], dic[j]
    return dic


def single_image(imagePath, p):
    """Main function.

    Args: 
        imagePath: The file path of image.
        p: The kousuanti rectangle region coordinate.

    Returns:
        Two dimensional array.

    Raises: 
        IOError: An error occurred accessing the imagePath object.

    """
    imgPIL = Image.open(imagePath)
    imageCV2 = np.array(imgPIL.convert('RGB'))
    kstImage = imgPIL.crop(tuple(p))
    position = get_template(imagePath, p)
    template = imgPIL.crop(tuple(position))
    #template.show()
    coordinate = template_match(kstImage, template, 0.7)
    #print ('coordinate = \n{}'.format(coordinate))
    #coorSort = sortPoint(coordinate)
    #print ('coorSort = \n{}'.format(coorSort))
    for i,j in coordinate.copy().items():
        coordinate[i] = [j[0]+p[0],j[1]+p[1],j[2]+p[0],j[3]+p[1]]

    result = {}
    for i,j in coordinate.items():
        x = 0
        if j[0]-400 >0 :
            cropImage = imgPIL.crop((j[0]-400, j[1], j[2], j[3]))
            #cropImage.show()
            img_rgb = np.array(cropImage.convert('RGB'))
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            bw = preprocess_bw_inv(img_gray)
            weight = vertical_proj(bw)
            rightBound = 0
            start = 0
            while sum(weight[start:start+50]) > 100 :
                start += 1

            for w in range(start,len(weight)) :
                if sum(weight[w:w+10]) > 100 :
                    rightBound = w
                    break

            #print (len(weight))
            #print ('weight = {}'.format(weight))
            #print ('rightBound = {}'.format(rightBound))
            #print ('position = {}'.format(position))
            x = j[0]-400+rightBound
        cropImage = imgPIL.crop((x, j[1], j[2], j[3]))
        rightBound = 0
        leftBound = 0
        img_rgb = np.array(cropImage.convert('RGB'))
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        bw = preprocess_bw_inv(img_gray)
        weight = vertical_proj(bw)
        for w in range(0,len(weight)) :
            if weight[w] != 0 :
                leftBound = w
                break
        for w in range(len(weight)-1,-1,-1):
            if weight[w] != 0 :
                rightBound = w
                break
        result[i] = [int(x+leftBound),int(j[1]), int(x+rightBound), int(j[3])]
        #cv2.rectangle(imageCV2,(result[0],result[1]),(result[2],result[3]),(0,255,0),2)
        #cv2.putText(imageCV2,str(i), (j[2]+10,j[3]), font, 1, 0, 1)
        #cv2.rectangle(imageCV2,(int(j[0]),int(j[1])),(int(j[2]),int(j[3])),(0,255,0),1)
    r = [[],[],[],[]]
    for i,j in result.items():
        r[0].append(j[0])
        r[1].append(j[1])
        r[2].append(j[2])
        r[3].append(j[3])
    return r





