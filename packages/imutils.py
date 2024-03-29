#!/usr/bin/env python

'''
图像工具
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
# import matplotlib.pyplot as plt
import numpy as np
import cv2
import warnings

PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range


def preprocess_bw_inv(gray, smooth=True):
    if smooth:
        gray = cv2.boxFilter(gray, -1, (3, 3))

    ret, bw = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw2 = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, element)

    # cv2.imshow('image1', bw2)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # bw2[bw2 == 255] = 1
    return bw2


def preprocess_bw(gray, smooth=True):
    if smooth:
        gray = cv2.boxFilter(gray, -1, (3, 3))

    ret, bw = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)  # 修改成235

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw2 = cv2.morphologyEx(bw, cv2.MORPH_OPEN, element)

    # cv2.imshow('image1', bw2)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # bw2[bw2 == 255] = 1
    return bw2


def first_any(arr, axis, target, invalid_val=-1):
    # 第一个任意值的位置
    mask = arr == target
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


# weight 是每一列的重量，profile 是每一列第一个非零点到最后一个非零点的距离
# profile -1 表示该列没有白点
def vertical_proj(bw):
    # img: black white
    [h, _] = bw.shape

    # 一个像素点权重算1，不算255
    bw = bw // 255
    weight = np.sum(bw, axis=0)
    nz1 = first_nonzero(bw, 0, -1)
    bw2ud = np.flipud(bw)
    nz2 = h - 1 - first_nonzero(bw2ud, 0, -1)
    nz2[nz2 == h] = -1
    profile = nz2 - nz1 + 1
    profile[nz1 == -1] = -1

    profile = profile.astype(np.float32)
    profile = cv2.GaussianBlur(profile, (1, 3), 0.8)[:, 0]
    return weight, profile


def nz_analysis(x):
    """分析非零的起始和长度"""
    length = len(x)
    zpos = np.nonzero(x == 0)

    nz_span = np.append(zpos, length) - np.append(-1, zpos) - 1
    nz_start = np.append(zpos, length) - nz_span

    return nz_start[nz_span != 0], nz_span[nz_span != 0]


# weight 是每一行的重量，profile 是每一行第一个非零点到最后一个非零点的距离
def horizontal_proj(bw):
    return vertical_proj(np.transpose(bw))


def align_images(im1, im_ref, savematch=False):
    # 需要调整下面参数，保证全部试卷可以，又提高速度
    MAX_FEATURES = 1000
    GOOD_MATCH_PERCENT = 0.5

    # Convert images to grayscale
    # if im1.shape[2]
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im_ref, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    if savematch:
        # Draw top matches
        imMatches = cv2.drawMatches(im1, keypoints1, im_ref, keypoints2, matches, None)
        cv2.imwrite("../result/matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    # h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # h, _ = cv2.estimateAffinePartial2D(points1, points2, method=cv2.RANSAC)
    h, _ = cv2.estimateAffine2D(points1, points2, method=cv2.RANSAC)

    # Use homography
    height, width, channels = im_ref.shape
    # im1Reg = cv2.warpPerspective(im1, h, (width, height))
    im1Reg = cv2.warpAffine(im1, h, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return im1Reg, h


def align_images2x2(im, imReference, savematch=False):
    h, w = im.shape[:2]
    href, wref = imReference.shape[:2]

    im11 = im[0:h // 2, 0:w // 2]
    imReference11 = imReference[0:href // 2, 0:wref // 2]

    im12 = im[0:h // 2:, w // 2:]
    imReference12 = imReference[0:href // 2, wref // 2:]

    im21 = im[h // 2:, 0:w // 2]
    imReference21 = imReference[href // 2:, 0:wref // 2]

    im22 = im[h // 2:, w // 2:]
    imReference22 = imReference[href // 2:, wref // 2:]

    imReg11, _ = align_images(im11, imReference11)
    imReg12, _ = align_images(im12, imReference12)
    imReg21, _ = align_images(im21, imReference21)
    imReg22, _ = align_images(im22, imReference22)

    imReg = np.empty_like(imReference)
    imReg[0:href // 2, 0:wref // 2] = imReg11
    imReg[0:href // 2, wref // 2:] = imReg12
    imReg[href // 2:, 0:wref // 2] = imReg21
    imReg[href // 2:, wref // 2:] = imReg22

    return imReg


def correct_skew(image, column=2):
    h, w = image.shape[:2]
    # 只针对双栏有调整
    sy = int(h * 0.05)
    ey = int(h * 0.19)
    sx = int(w * 0.08)
    ex = int(w * 0.47)

    if column == 1:
        warnings.warn('单栏尚未实现')
        return image, 0

    head = image[sy:ey, sx:ex]

    # imu.imshow(head)

    # convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    gray = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)

    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # # draw the correction angle on the image so we can validate it
    # cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
    #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return rotated, angle


def imshow_(img, mode=cv2.WINDOW_NORMAL):
    cv2.namedWindow("temp", mode)
    cv2.imshow("temp", img)
    cv2.waitKey()
    cv2.destroyWindow("temp")
