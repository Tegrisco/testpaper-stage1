from __future__ import print_function
import cv2
import numpy as np
import imutils as imu
from PIL import ImageChops
from PIL import Image
import os
import preprocess

# def separate(empty, answered):
#
#     diff1 = cv2.absdiff(empty, answered)
#     # imu.imshow_(diff1)
#
#
#     diff1 = cv2.cvtColor(diff1, cv2.COLOR_BGR2GRAY)
#
#
#     diff2 = cv2.cvtColor(empty, cv2.COLOR_BGR2GRAY) - cv2.cvtColor(answered, cv2.COLOR_BGR2GRAY)
#     # imu.imshow_(diff2)
#     seperate_img = (diff1 | diff2) - (diff1 ^ diff2)
#     imu.imshow_(255-seperate_img)
#
#     # imu.imshow_(seperate_img)
#
#
#     # kernel = np.ones((4, 4), np.uint8)  # 设置卷积核的大小，可控制腐蚀的程度
#     # erosion = cv2.erode(seperate_img, kernel)
#     # GaussBlur = cv2.GaussianBlur(erosion, (11, 11), 1.2)  # 高斯去噪，消除字体锯齿
#     # ret, binary = cv2.threshold(GaussBlur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_TRIANGLE)
#
#
#     # ret, binary = cv2.threshold(seperate_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
#     ret, binary = cv2.threshold(seperate_img, 0, 255, cv2.THRESH_OTSU)
#     # imu.imshow_(binary)
#     kernel = np.ones((3, 3), np.uint8)  # 设置卷积核的大小，可控制腐蚀的程度
#     # b2 = cv2.erode(binary, kernel)
#
#     b3 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
#
#     # kernel2 = np.ones((7, 7), np.uint8)  # 设置卷积核的大小，可控制腐蚀的程度
#     # b3 = cv2.morphologyEx(b2, cv2.MORPH_CLOSE, kernel2)
#
#     # GaussBlur = cv2.GaussianBlur(b3, (5, 5), -1)  # 高斯去噪，消除字体锯齿
#
#     # GaussBlur = 255 - GaussBlur
#
#     b3 = 255 - b3
#
#     return b3


def separate2(empty_gray, answered_gray):
    imReference = imu.preprocess_bw(empty_gray)
    aligned = imu.preprocess_bw(answered_gray)

    imstack = cv2.addWeighted(imReference, 0.7, aligned, 0.3, 0)
    # imu.imshow_(imstack)
    # imstack_gray = cv2.cvtColor(imstack, cv2.COLOR_BGR2GRAY)
    pos = imstack > 200
    imstack[pos] = np.bitwise_and(imstack[pos], 0)

    # imu.imshow_(imstack)

    ret, imstack = cv2.threshold(imstack, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 4))
    imstack = cv2.morphologyEx(imstack, cv2.MORPH_CLOSE, element)

    return imstack


# def test_all():
#     srcdir = '../img/paper/empty'
#     srcdir2 = '../result/aligned'
#
#     outdir = '../result/separate'
#
#     papers = os.listdir(srcdir)
#     result = open(os.path.join(outdir, "list.txt"),'w')
#     for paper in papers:
#         if paper.endswith('.jpg'):
#             refFilename = os.path.join(srcdir, paper)
#             print("母卷", refFilename)
#             imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
#
#             imFilename = os.path.join(srcdir2, paper)
#             print("答卷", imFilename)
#             aligned = cv2.imread(imFilename, cv2.IMREAD_COLOR)
#
#             result = separate(imReference, aligned)
#
#             cv2.imwrite(os.path.join(outdir, paper[:-4] + '-ans.jpg'), result)

def test_all2():
    srcdir = '../img/paper/empty'
    srcdir2 = '../result/aligned'

    outdir = '../result/separate2'

    papers = os.listdir(srcdir)
    result = open(os.path.join(outdir, "list.txt"), 'w')
    for paper in papers:
        if paper.endswith('.jpg'):
            refFilename = os.path.join(srcdir, paper)
            # print("母卷", refFilename)
            imReference = cv2.imread(refFilename, 0)

            imFilename = os.path.join(srcdir2, paper)
            # print("答卷", imFilename)
            aligned = cv2.imread(imFilename, 0)

            result = separate2(imReference, aligned)

            cv2.imwrite(os.path.join(outdir, paper[:-4] + '-ans.jpg'), result)


# def single():
#     srcdir = '../img/paper/empty'
#     srcdir2 = '../result/aligned'
#
#     outdir = '../result/separate'
#
#     paper = '6-2.jpg'
#     refFilename = os.path.join(srcdir, paper)
#     print("母卷", refFilename)
#     imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
#
#     imFilename = os.path.join(srcdir2, paper)
#     print("答卷", imFilename)
#     aligned = cv2.imread(imFilename, cv2.IMREAD_COLOR)
#
#     result = separate(imReference, aligned)
#
#     cv2.imwrite(os.path.join(outdir, paper[:-4]+'-ans.jpg'), result)


def single2():
    # srcdir = '../img/paper/empty'
    # srcdir2 = '../result/aligned'
    #
    # outdir = '../result/separate2'
    #
    # paper = '6-2.jpg'
    # refFilename = os.path.join(srcdir, paper)
    # print("母卷", refFilename)
    # # imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
    # imReference = cv2.imread(refFilename, 0)
    #
    # imFilename = os.path.join(srcdir2, paper)
    # print("答卷", imFilename)
    # # aligned = cv2.imread(imFilename, cv2.IMREAD_COLOR)
    # aligned = cv2.imread(imFilename, 0)
    imReference = cv2.imread('D://exam_img/real_image/full/1-1.jpg', cv2.IMREAD_COLOR)
    imReference = cv2.cvtColor(imReference,cv2.COLOR_RGB2GRAY)
    # aligned = cv2.imread('D://exam_img/answer_image/full/1-1.jpg', cv2.IMREAD_COLOR)
    aligned = preprocess.align('D://exam_img/real_image/full/1-1.jpg', 'D://exam_img/answer_image/full/1-1.jpg')
    aligned = cv2.cvtColor(aligned,cv2.COLOR_RGB2GRAY)
    result = separate2(imReference, aligned)

    cv2.imwrite('./imgStore/separate.jpg', result)


if __name__ == '__main__':
    # single()
    # test_all()

    single2()
    # test_all2()
