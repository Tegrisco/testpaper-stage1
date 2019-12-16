"""
=====================================
Get the coordinates of the student ID

Author: Zheng Zhihuang

Date: 2019.7.12
=====================================

A script for Python3.

Use template matching to locate the XueHao region.
使用模版匹配的方法定位学号区域。

"""


# Use matchTemplate function of opencv
import cv2 as cv

# Standard scientific Python imports
import numpy as np

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def xuehao(imagePath):
    """Fetches the position of XueHao region from a paper image.

    Use prepared template to get the first match position in image.
    The result is precise, but it may not apply to some kind of 
    papers with different fonts because the template is changeless
    and the size of region is fixed (260*83pixel) too.

    Args:
        imagePath: The file path of image.

    Returns:
        An array with four integer elements includes the coordinates of 
        the upper left corner and the lower right corner of the xuehao 
        region. For example:

        [1, 2, 3, 4]

    Raises:
        IOError: An error occurred accessing the imagePath object.

    """
    imgRgb = cv.imread(imagePath)
    imgGray = cv.cvtColor(imgRgb, cv.COLOR_BGR2GRAY)
    template = cv.imread(os.path.join(BASE_DIR, 'packages/xuehao_templates/xuehao.jpg'), 0)
    w, h = template.shape[::-1]
    res = cv.matchTemplate(imgGray, template, cv.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    idPosition = []
    for pt in zip(*loc[::-1]):
        idPosition = [pt[0]+w, pt[1]-h/2, pt[0]+w+260, pt[1]+h]
        break
    return [int(x) for x in idPosition]

