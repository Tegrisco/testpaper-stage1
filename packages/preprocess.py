import imutils as imu
import cv2
import numpy as np
import separate_ans2


def align(refImage, alignedImage):
    """
    align(refImage, alignedImage):通过调用imutil的对齐函数获取对齐试卷
    :param refImage: 母卷路径
    :param alignedImage: 答案卷路径
    :return: 对齐之后的试卷(opencv格式)
    """
    ref = cv2.imread(refImage, cv2.IMREAD_COLOR)
    aligned = cv2.imread(alignedImage, cv2.IMREAD_COLOR)
    imReg = imu.align_images2x2(aligned, ref)
    return imReg


def separateAns(refImagePath, alighedImagePath):
    """
    separateAns(refImagePath, alighedImagePath):通过调用分离代码获取分离试卷
    :param refImagePath: 母卷路径
    :param alighedImagePath: 答案卷路径
    :return: 分离之后的试卷（opencv格式）
    """
    imReference = cv2.imread(refImagePath, cv2.IMREAD_COLOR)
    aligned = align(refImagePath, alighedImagePath)
    imReference = cv2.cvtColor(imReference, cv2.COLOR_RGB2GRAY)
    aligned = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY)
    result = separate_ans2.separate2(imReference, aligned)
    return result

