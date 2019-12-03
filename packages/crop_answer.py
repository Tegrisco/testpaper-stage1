#!/usr/bin/env python

'''
This program illustrates the use of findContours and drawContours.
The original image is put up along with the image of drawn contours.

Usage:
    contours.py
A trackbar is put up which controls the contour level from -3 to 3
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys

PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2
import os
import json
import imutils as imu


def reduce_margin(bw_inv, profile=None, hori_vert=0):
    if profile is None:
        if hori_vert == 0:
            _, profile = imu.vertical_proj(bw_inv)
        else:
            _, profile = imu.horizontal_proj(bw_inv)

    profile_nz = profile > 0
    nzx1 = imu.first_nonzero(profile_nz, 0, -1) - 0
    nzx2 = profile.shape[0] - imu.first_nonzero(np.flip(profile_nz, axis=0), 0, -1)

    return (nzx1, nzx2)


def giveme_answer(paper_gray, sep_gray, box, type):
    """

    :param paper_gray:
    :param sep_gray:
    :param box:
    :param type:
    :return:
    """
    empty_thresh = 0.01
    hori_margin = 20  # in pixel
    vert_margin_ratio = 1.0  # in ratio
    vert_margin_conservative = 15  # 在判断左右时，上下先延展这个数值

    # type 0 方框，圆圈
    if type == 0:
        hori_margin = 5  # in pixel
        vert_margin_ratio = 0.3  # in ratio
        vert_margin_conservative = 5  # 在判断左右时，上下先延展这个数值

    sep_gray_inv = 255 - sep_gray

    x1, y1, x2, y2 = box
    x2 = x2 + 1
    y2 = y2 + 1
    w = x2 - x1
    h = y2 - y1

    # if x1 < 515 and x2 > 515 and y1 < 1743 and y2 > 1743:
    #     imu.imshow_(paper_gray[y1:y2,x1:x2])

    empty = False

    # 判断是否是空白
    if cv2.countNonZero(sep_gray_inv[y1:y2, x1:x2]) / float(w * h) < empty_thresh:
        empty = True
        return box, empty

    # type 0 方框，圆圈
    # if type == 0:
    #     return box, empty

    # 左右延展
    x1 = x1 - hori_margin
    x2 = x2 + hori_margin

    # 上下延展

    y1_tmp = y1 - vert_margin_conservative
    y2_tmp = y2 + vert_margin_conservative

    # 垂直分析
    weight, profile = imu.vertical_proj(sep_gray_inv[y1_tmp:y2_tmp, x1:x2])

    # if x1 < 515 and x2 > 515 and y1 < 1743 and y2 > 1743:
    #     imu.imshow_(sep_gray_inv[y1_tmp:y2_tmp,x1:x2], cv2.WINDOW_AUTOSIZE)

    # 左边第一个非零，右边第一个非零
    profile_nz = profile > 0
    profile_nz = np.logical_and(profile_nz, weight > 2)
    nzx1 = imu.first_nonzero(profile_nz, 0, -1) - 0
    nzx2 = x2 - x1 - imu.first_nonzero(np.flip(profile_nz, axis=0), 0, -1)

    # horizontal best
    x1b = x1 + nzx1
    x2b = x1 + nzx2

    vert_margin = int(round(h * vert_margin_ratio))
    y1 = y1 - vert_margin
    y2 = y2 + vert_margin
    half_height = (y2 - y1) // 2

    # if x1 < 515 and x2 > 515 and y1 < 1743 and y2 > 1743:
    #     imu.imshow_(sep_gray_inv[y1:y2, x1b:x2b],cv2.WINDOW_AUTOSIZE)

    weight, profile = imu.horizontal_proj(sep_gray_inv[y1:y2, x1b:x2b])
    # 上面延展部分的第一行空白（由下往上）
    profile_z_true = profile == -1
    temp = imu.first_nonzero(np.flip(profile_z_true[:vert_margin], axis=0), 0, -1)
    if temp == -1:  #
        y1b = y1
    else:
        y1b = y1 + (vert_margin - temp)
    # 下面延展部分的第一行空白（由上往下）
    temp = imu.first_nonzero(profile_z_true[h + vert_margin:], 0, -1)
    if temp == -1:
        y2b = y2
    else:
        y2b = y2 - vert_margin + temp

    y1b_delta, y2b_delta = reduce_margin(None, profile[y1b - y1:y2b - y1], 1)
    anchor = y1b
    y1b = anchor + y1b_delta
    y2b = anchor + y2b_delta

    return (x1b, y1b, x2b, y2b), empty


def main():
    srcdir = '../result/aligned'
    # sepdir = '../result/separate2'
    sepdir = '../result/separate3'
    jsondir = '../result/pos-json'
    outdir = '../result/cut'

    papers = os.listdir(srcdir)
    # papers = ['2-1.jpg']

    result = open(os.path.join(outdir, "list.txt"), 'w')
    for paper in papers:
        if paper.endswith('.jpg') and not paper[:-4].endswith('stack'):

            trunk = os.path.splitext(paper)[0]

            src = cv2.imread(os.path.join(srcdir, paper))
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

            sepgray = cv2.imread(os.path.join(sepdir, trunk + '-ans.jpg'), 0)
            sep = cv2.cvtColor(sepgray, cv2.COLOR_GRAY2BGR)

            # 处理json
            json_path = os.path.join(jsondir, trunk + '.json')
            boxes = json.load(open(json_path, 'r'))
            boxes = np.array(boxes).astype(np.int)

            # 取高中位值作为行高
            raw_height = np.median(boxes[:, 3] - boxes[:, 1])
            underline_idx = boxes[:, 5] == 2
            print(boxes[underline_idx, :])
            boxes[underline_idx, 1] = boxes[underline_idx, 1] - raw_height
            print(boxes[underline_idx, :])

            for box in boxes:

                rect, is_empty = giveme_answer(gray, sepgray, box[:4], box[5])
                x1, y1, x2, y2 = rect
                x2 = x2 - 1
                y2 = y2 - 1

                # font = cv2.FONT_HERSHEY_SIMPLEX
                # height = ey - sy + 1
                # cv2.rectangle(src, (sx-10, sy-height), (ex+10, ey+height), (0, 0, 255), 3)
                # cv2.rectangle(sep, (sx-10, sy-height), (ex+10, ey+height), (0, 0, 255), 3)

                if box[5] == 0:
                    cv2.rectangle(src, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.rectangle(sep, (x1, y1), (x2, y2), (0, 255, 0), 1)
                else:
                    cv2.rectangle(src, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.rectangle(sep, (x1, y1), (x2, y2), (0, 0, 255), 1)
                # cv2.putText(src, str(idx), (sx, sy), font, 4, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(outdir, paper), src)
            cv2.imwrite(os.path.join(outdir, trunk + '-ans.jpg'), sep)

    result.close()


if __name__ == '__main__':
    print(__doc__)
    main()
    cv2.destroyAllWindows()
