# 作者：简卓林
# 日期：2019.7.12
# Version:1.0.0

# from tesserocr import PyTessBaseAPI, RIL, Image
import cv2
import copy
import preprocess
import imutils
import handleTextLine
# Ad-doc solution for running Tesseract OCR on a mixing locale environment
import locale
import numpy as np
from contextlib import contextmanager
import os


class calAnsPos(object):
    @contextmanager
    def c_locale(self):
        """
         c_locale(self):设置语言
        :return:
        """
        try:
            currlocale = locale.getlocale()
        except ValueError:
            currlocale = ('en_US', 'UTF-8')
        # print ('Switching to C from {}'.format(currlocale))
        locale.setlocale(locale.LC_ALL, "C")
        yield
        # print ('Switching to {} from C'.format(currlocale))
        locale.setlocale(locale.LC_ALL, currlocale)

    def __init__(self, qPosition, AnsImg, modelImage):
        """
        __init__(self, qPosition, AnsImg, modelImage):初始化数据为了给后面的答案区选出函数使用
        :param qPosition: 口算题大题的区域坐标[x0,y0,x1,y1]
        :param AnsImg: 分离卷
        :param modelImage: 母卷
        """

        def ans_scan(ansImage, levelUsage, qPosition):
            """
            ans_scan(ansImage, levelUsage, qPosition):tesseract扫描图片函数
            :param ansImage:答案卷
            :param levelUsage:tesseract的level
            :param qPosition:口算题大题的区域坐标[x0,y0,x1,y1]
            :return:基于symbol的box{x,y,w,h}
            """
            pil_im = Image.fromarray(ansImage)
            with PyTessBaseAPI(psm=6, lang="chi_sim") as api:
                api.SetImage(pil_im)
                api.SetRectangle(qPosition[0], qPosition[1], qPosition[2] - qPosition[0],
                                 qPosition[3] - qPosition[1])
                boxes = api.GetComponentImages(levelUsage, True)
            return boxes

        with self.c_locale():
            from tesserocr import RIL, PyTessBaseAPI, Image
            self.boxes_answer = ans_scan(AnsImg, RIL.SYMBOL, qPosition)
            self.sepAnsImg = AnsImg
            self.handledTextLine = handleTextLine.getHandleTextLine(modelImage)

    def cropAns(self, ansPostion, oralBox, i):
        """
        cropAns(self, ansPostion, oralBox, i):用于裁剪白边
        :param ansPostion: 基于答案区选出算法后得到的box[x0,y0,x1,y1]
        :param oralBox: 基于母卷扫描后得到的口算题题干的坐标
        :param i: 口算题所属的题目的序号
        :return: 裁取白边后得到的新答案box[x0,y0,x1,y1]
        """

        def searchWhichTextline(orallBox, handledTextLine):
            """
            searchWhichTextline(orallBox, handledTextLine):获取答案所在行的行高
            :param orallBox: 基于母卷扫描后得到的口算题题目的坐标[x0,y0,x1,y1]
            :param handledTextLine: 母卷(双栏)每一行的行高
            :return: 答案区所在的行区域item{x,y,w,h}
            """
            # print("handleAll", handledTextLine)
            for item in handledTextLine:
                # print(item)
                if item['x'] <= orallBox[0] <= orallBox[2] <= item['x'] + item['w'] and item['y'] <= orallBox[1] <= \
                        orallBox[3] <= item['y'] + item['h']:
                    return item
            return

        def topImpurityCut(bw):
            """
            topImpurityCut(bw):清除上边白边
            :param bw:基于答案区选出算法后得到的答案裁剪出来的二值图（反色处理）
            :return:上边白边的所需裁剪的像素值
            """
            weight, profileH = imutils.horizontal_proj(bw)
            topSub = imutils.first_any(profileH, 0, -1)
            return topSub

        def bottomImpurityCut(bw):
            """
            bottomImpurityCut(bw):清除下边白边
            :param bw:基于答案区选出算法后得到的答案裁剪出来的二值图（反色处理）
            :return:下边白边的所需裁剪的像素值
            """
            weight, profileH = imutils.horizontal_proj(bw)
            bottomSub = imutils.first_any(np.flip(profileH), 0, -1)
            return bottomSub

        def leftRightCut(profileV):
            """
            leftRightCut(profileV):左右白边裁剪
            :param profileV: 答案区反色处理后垂直投影（即：每列的像素总量）
            :return: 左右白边的所需裁剪的像素值（int)
            """
            leftRelativePosMax = imutils.first_nonzero(profileV > 10, 0)
            # print('np.flip(profileV[:leftRelativePosMax]):', np.flip(profileV[:leftRelativePosMax]))
            if leftRelativePosMax > 0:
                leftRelativePosSub = imutils.first_any(np.flip(profileV[:leftRelativePosMax]), 0, -1)
                if leftRelativePosSub == -1:
                    leftRelativePosSub = leftRelativePosMax
                # print("leftRelativePosSub:", leftRelativePosSub)
                leftRelativePos = leftRelativePosMax - leftRelativePosSub
            else:
                leftRelativePos = leftRelativePosMax
            rightRelativePosMax = imutils.first_nonzero(np.flip(profileV) > 10, 0)
            if rightRelativePosMax > 0:
                rightRelativeSub = imutils.first_any(profileV[len(profileV) - rightRelativePosMax:], 0, -1)
                if rightRelativeSub == -1:
                    rightRelativeSub = rightRelativePosMax
                rightRelativePos = rightRelativePosMax - rightRelativeSub
            else:
                rightRelativePos = rightRelativePosMax
            return leftRelativePos, rightRelativePos

        def upBottomCut(profileH):
            """
            upBottomCut(profileH):上下白边裁剪
            :param profileH: 答案区反色处理后水平投影（即：每行的像素总量）
            :return: 上下白边的所需裁剪的像素值（int)
            """
            upRelativeMax = imutils.first_nonzero(profileH > 10, 0)
            if upRelativeMax > 0:
                upRelativeSub = imutils.first_any(np.flip(profileH[:upRelativeMax]), 0, -1)
                if upRelativeSub == -1:
                    upRelativeSub = upRelativeMax
                upRelativePos = upRelativeMax - upRelativeSub
            else:
                upRelativePos = upRelativeMax
            bottomRelativeMax = imutils.first_nonzero(np.flip(profileH) > 10, 0)
            if bottomRelativeMax > 0:
                bottomRelativeSub = imutils.first_any(profileH[len(profileH) - bottomRelativeMax:], 0, -1)
                if bottomRelativeSub == -1:
                    bottomRelativeSub = bottomRelativeMax
                bottomRelativePos = bottomRelativeMax - bottomRelativeSub
            else:
                bottomRelativePos = bottomRelativeMax
            return upRelativePos, bottomRelativePos

        try:
            handledTextLine = searchWhichTextline(oralBox, self.handledTextLine)
            if handledTextLine:
                # print('i:', i)
                if ansPostion[1] < handledTextLine['y']:
                    ansPostion[1] = handledTextLine['y']
                    bw = 255 - self.sepAnsImg[ansPostion[1]:ansPostion[3], ansPostion[0]:ansPostion[2]]
                    topSub = topImpurityCut(bw)
                    # print('topSub:', topSub)
                    ansPostion[1] = ansPostion[1] + topSub
                if ansPostion[3] > handledTextLine['y'] + handledTextLine['h']:
                    ansPostion[3] = handledTextLine['y'] + handledTextLine['h']
                    bw = 255 - self.sepAnsImg[ansPostion[1]:ansPostion[3], ansPostion[0]:ansPostion[2]]
                    bottomSub = bottomImpurityCut(bw)
                    # print('bottomSub:', bottomSub)
                    ansPostion[3] = ansPostion[3] - bottomSub
            bw = 255 - self.sepAnsImg[ansPostion[1]:ansPostion[3], ansPostion[0]:ansPostion[2]]
            # if i == 46:
            #     cv2.imwrite('template/test.jpg', bw)
            #     cv2.imwrite('template/unbw.jpg',unbw)
            weight, profileV = imutils.vertical_proj(bw)
            leftRelativePos, rightRelativePos = leftRightCut(profileV)
            weight, profileH = imutils.horizontal_proj(bw)
            upRelativePos, bottomRelativePos = upBottomCut(profileH)
            # if i == 46:
            #     plt.plot(profileV)
            #     plt.title(i)
            #     plt.show()
            ansPostion[0] = ansPostion[0] + leftRelativePos
            ansPostion[1] = ansPostion[1] + upRelativePos
            ansPostion[2] = ansPostion[2] - rightRelativePos
            ansPostion[3] = ansPostion[3] - bottomRelativePos
        except Exception:
            return ansPostion
        return ansPostion

    def answer_zone(self, position, Q_position):
        """
        answer_zone(self, position, Q_position):获取口算题答案区的坐标
        :param position: 口算题题目区域
        :param Q_position: 口算题大题坐标[x0,y0,x1,y1]
        :return: 口算题区域答案区坐标
        """

        def preUseCropAns(x1, y1, x2, y2):
            """
            preUseCropAns(x1, y1, x2, y2):口算题答案区数据结构修改
            :param x1: 答案区x0坐标
            :param y1: 答案区y0坐标
            :param x2: 答案区x1坐标
            :param y2: 答案区y1坐标
            :return:新数据结构的口算题答案区坐标[x0,y0,x1,y1]
            """
            ansPos = []
            ansPos.append(x1)
            ansPos.append(y1)
            ansPos.append(x2)
            ansPos.append(y2)
            return ansPos

        # 左上角
        def detectUpLeft(boxes_answer, position, Q_position, i, j, plus):
            """
            detectUpLeft(boxes_answer, position, Q_position, i, j, plus):检测tesseract扫描的口算题区域是否在指定答案区域内
            :param boxes_answer: 基于tesseract扫描答案卷口算题区域得到的boxes[{x,y,w,h}]
            :param position: 口算题题目的区域
            :param Q_position: 口算题大题的区域
            :param i: 口算题序号
            :param j: 需要检测的symbol的index
            :param plus: 差值
            :return:1:symbol在指定答案区域 0:symbol未在指定答案区域
            """
            if (position[0][i] < boxes_answer[j][1]['x'] + Q_position[0] < position[0][i + 1]
                    and (boxes_answer[j][1]['y'] + boxes_answer[j][1]['h'] + Q_position[1]) <
                    position[1][i + plus + 1] + 10 and boxes_answer[j][1]['y'] + Q_position[1] >
                    Q_position[1]):
                return 1
            else:
                return 0

        # 右上角
        def detectUpRight(boxes_answer, position, Q_position, i, j, plus):
            """
            detectUpRight(boxes_answer, position, Q_position, i, j, plus):检测tesseract扫描的口算题区域是否在指定答案区域内
            :param boxes_answer: 基于tesseract扫描答案卷口算题区域得到的boxes[{x,y,w,h}]
            :param position: 口算题题目的区域
            :param Q_position: 口算题大题的区域
            :param i: 口算题序号
            :param j: 需要检测的symbol的index
            :param plus: 差值
            :return:1:symbol在指定答案区域 0:symbol未在指定答案区域
            """
            if (position[0][i] < boxes_answer[j][1]['x'] + Q_position[0] < Q_position[2] and (
                    boxes_answer[j][1]['y'] + boxes_answer[j][1]['h'] + Q_position[1]) <
                    position[1][i + plus + 1] + 10 and boxes_answer[j][1]['y'] + Q_position[1] >
                    Q_position[1] - 10
                    and boxes_answer[j][1]['x'] + boxes_answer[j][1]['w'] + Q_position[0] < Q_position[
                        2]
            ):
                return 1
            else:
                return 0

        # 左下角
        def detectLowLeft(boxes_answer, position, Q_position, i, j, plus):
            """
            detectLowLeft(boxes_answer, position, Q_position, i, j, plus):检测tesseract扫描的口算题区域是否在指定答案区域内
            :param boxes_answer: 基于tesseract扫描答案卷口算题区域得到的boxes[{x,y,w,h}]
            :param position: 口算题题目的区域
            :param Q_position: 口算题大题的区域
            :param i: 口算题序号
            :param j: 需要检测的symbol的index
            :param plus: 差值
            :return:1:symbol在指定答案区域 0:symbol未在指定答案区域
            """
            if (position[0][i] < boxes_answer[j][1]['x'] + Q_position[0] < position[0][i + 1] and
                    boxes_answer[j][1][
                        'y'] +
                    Q_position[1] >
                    position[3][i - plus - 1] and boxes_answer[j][1]['y'] + boxes_answer[j][1]['h'] +
                    Q_position[1] <
                    Q_position[3]
                    and boxes_answer[j][1]['x'] + boxes_answer[j][1]['w'] + Q_position[0] < Q_position[2]
            ):
                return 1
            else:
                return 0

        # 右下角
        def detectLowRight(boxes_answer, position, Q_position, i, j, plus):
            """
            detectLowRight(boxes_answer, position, Q_position, i, j, plus):检测tesseract扫描的口算题区域是否在指定答案区域内
            :param boxes_answer: 基于tesseract扫描答案卷口算题区域得到的boxes[{x,y,w,h}]
            :param position: 口算题题目的区域
            :param Q_position: 口算题大题的区域
            :param i: 口算题序号
            :param j: 需要检测的symbol的index
            :param plus: 差值
            :return:1:symbol在指定答案区域 0:symbol未在指定答案区域
            """
            if (position[0][i] < boxes_answer[j][1]['x'] + Q_position[0] < Q_position[2] and boxes_answer[j][1]['y'] +
                    Q_position[1] >
                    position[3][i - plus - 1] and boxes_answer[j][1]['y'] + boxes_answer[j][1][
                        'h'] + Q_position[1] < Q_position[3]):
                return 1
            else:
                return 0

        # 中部右边，底下无题
        def detectMiddleRightNoSub(boxes_answer, position, Q_position, i, j, plus):  # 解决倒二层楼下没有东西的问题
            """
            detectMiddleRightNoSub(boxes_answer, position, Q_position, i, j, plus):检测tesseract扫描的口算题区域是否在指定答案区域内
            :param boxes_answer: 基于tesseract扫描答案卷口算题区域得到的boxes[{x,y,w,h}]
            :param position: 口算题题目的区域
            :param Q_position: 口算题大题的区域
            :param i: 口算题序号
            :param j: 需要检测的symbol的index
            :param plus: 差值
            :return:1:symbol在指定答案区域 0:symbol未在指定答案区域
            """
            if (position[0][i] < boxes_answer[j][1]['x'] + Q_position[0] < Q_position[2]
                    and boxes_answer[j][1]['y'] + Q_position[1] > position[1][i - plus - 1]
                    and boxes_answer[j][1]['y'] + boxes_answer[j][1]['h'] + Q_position[1] < Q_position[3]):
                return 1
            else:
                return 0

        # 中部右边，底下有题
        def detectMiddleRightSub(boxes_answer, position, Q_position, i, j, plus):
            """
            detectMiddleRightSub(boxes_answer, position, Q_position, i, j, plus):检测tesseract扫描的口算题区域是否在指定答案区域内
            :param boxes_answer: 基于tesseract扫描答案卷口算题区域得到的boxes[{x,y,w,h}]
            :param position: 口算题题目的区域
            :param Q_position: 口算题大题的区域
            :param i: 口算题序号
            :param j: 需要检测的symbol的index
            :param plus: 差值
            :return:1:symbol在指定答案区域 0:symbol未在指定答案区域
            """
            # if(position[0][i] < boxes_answer[j][1]['x'] + Q_position[0] < Q_position[2]):
            #     if(i == 5):
            #         print("detectx2:", boxes_answer[j][1]['x'] + boxes_answer[j][1]['w'] + Q_position[0])
            if (position[0][i] < boxes_answer[j][1]['x'] + Q_position[0] < Q_position[2] and (
                    boxes_answer[j][1]['y'] + boxes_answer[j][1]['h'] + Q_position[1]) <
                    position[1][i + plus + 1] + 15):
                return 1
            else:
                return 0

        # 中部左边，底下无题
        def detectMiddleLeftNoSub(boxes_answer, position, Q_position, i, j, plus):
            """
            detectMiddleLeftNoSub(boxes_answer, position, Q_position, i, j, plus):检测tesseract扫描的口算题区域是否在指定答案区域内
            :param boxes_answer: 基于tesseract扫描答案卷口算题区域得到的boxes[{x,y,w,h}]
            :param position: 口算题题目的区域
            :param Q_position: 口算题大题的区域
            :param i: 口算题序号
            :param j: 需要检测的symbol的index
            :param plus: 差值
            :return: 1:symbol在指定答案区域 0:symbol未在指定答案区域
            """
            if (position[0][i] < boxes_answer[j][1]['x'] + Q_position[0] < position[0][i + 1] and (
                    boxes_answer[j][1]['y'] + boxes_answer[j][1]['h'] + Q_position[1]) < Q_position[
                3] and
                    boxes_answer[j][1]['y'] + Q_position[1] > position[1][i - plus - 1]):
                return 1
            else:
                return 0

        # 中部左边，底下有题
        def detectMiddleLeftSub(boxes_answer, position, Q_position, i, j, plus):
            """
            detectMiddleLeftSub(boxes_answer, position, Q_position, i, j, plus):检测tesseract扫描的口算题区域是否在指定答案区域内
            :param boxes_answer: 基于tesseract扫描答案卷口算题区域得到的boxes[{x,y,w,h}]
            :param position: 口算题题目的区域
            :param Q_position: 口算题大题的区域
            :param i: 口算题序号
            :param j: 需要检测的symbol的index
            :param plus: 差值
            :return: 1:symbol在指定答案区域 0:symbol未在指定答案区域
            """
            if (position[0][i] < boxes_answer[j][1]['x'] + Q_position[0] < position[0][i + 1] and (
                    boxes_answer[j][1]['y'] + boxes_answer[j][1]['h'] + Q_position[1]) <
                    position[1][i + plus + 1] + 15 and
                    boxes_answer[j][1]['y'] + Q_position[1] > position[3][i - plus - 1]):
                return 1
            else:
                return 0

        # 获取答案区坐标
        def cropAns(x1List, y1List, x2List, y2List, Q_position, position, i, num, plus, type, temp1):
            """
            cropAns(x1List, y1List, x2List, y2List, Q_position, position, i, num, plus, type, temp1):汇总指定答案区域的symbol，得到答案区的左边界，有边界，上边界，下边界
            :param tempx1:答案区内含有的symbol的x0汇总后的列表
            :param tempy1:答案区内含有的symbol的y0汇总后的列表
            :param tempx2:答案区内含有的symbol的x1汇总后的列表
            :param tempy2:答案区内含有的symbol的y1汇总后的列表
            :param Q_position:口算题大题的区域
            :param position:口算题题目的区域
            :param i: 口算题序号
            :param num:口算题每行最右边的index
            :param plus:差值
            :param type:答案所处的区域（'up':最上层，'middle';中间层，'sub':最底层）
            :param temp1:目前可得到的答案区域
            :return:口算题指定区域答案区的坐标{x1,y1,x2,y2}
            """
            dict = {}
            try:
                if x1List:
                    if min(x1List) <= position[2][i]:
                        dict['x1'] = position[2][i]
                    else:
                        dict['x1'] = min(x1List)
                    # dict['x1'] = min(x1List)

                else:
                    dict['x1'] = position[2][i]
                    dict['y1'] = temp1[i - plus - 1][0]['y2']
                    dict['y2'] = position[1][i + plus + 1]
                    dict['x2'] = Q_position[2]
                dict['y1'] = min(y1List)
                dict['y2'] = max(y2List)
                dict['x2'] = max(x2List)

            except Exception:
                if type == 'up':
                    if i == num:
                        dict['x1'] = position[2][i]
                        dict['y1'] = Q_position[1]
                        dict['y2'] = position[1][i + plus + 1]
                        dict['x2'] = Q_position[2]
                    else:
                        dict['x1'] = position[2][i]
                        dict['y1'] = Q_position[1]
                        dict['y2'] = position[1][i + plus + 1]
                        dict['x2'] = position[0][i + 1]
                elif type == 'sub':
                    if i == num:
                        dict['y1'] = temp1[i - plus - 1][0]['y2']
                        dict['y2'] = Q_position[3]
                        dict['x2'] = Q_position[2]
                        dict['x1'] = position[2][i]
                    else:
                        dict['x1'] = position[2][i]
                        dict['y1'] = temp1[i - plus - 1][0]['y2']
                        dict['y2'] = Q_position[3]
                        dict['x2'] = position[0][i + 1]
                elif type == 'middle':
                    if i == num:
                        dict['x1'] = position[2][i]
                        dict['y1'] = temp1[i - plus - 1][0]['y2']
                        # print("有正常赋值dict['y1']:{}".format(dict['y1']))
                        if i + plus + 1 < len(position[0]):
                            dict['y2'] = position[1][i + plus + 1]
                        else:
                            dict['y2'] = Q_position[3]
                        dict['x2'] = Q_position[2]
                    else:
                        dict['x1'] = position[2][i]
                        dict['y1'] = temp1[i - plus - 1][0]['y2']
                        if i + plus + 1 < len(position[0]):
                            dict['y2'] = position[1][i + plus + 1]
                        else:
                            dict['y2'] = Q_position[3]
                        dict['x2'] = position[0][i + 1]
            return dict

        ques_num = []  # 题目行列
        y1List = []  # 临时存放y1的值，用于比较上边界
        y2List = []  # 临时存放y2的值，用于比较出下边界
        x1List = []  # 暂时存储x2的值，用于比较
        x2List = []  # 最终保存的list
        temp1 = []  # 最终保存的list
        temp2 = []  # 暂存list
        index = []  # 题目序号
        dict = {}  # 临时存放x0,y0,x1,y1的字典
        remove_item = []  # 可删除的答案索引
        space = []  # 存放未用的题目
        # print(position)
        # print(len(position[0]))
        i = 0

        # 对口算题题干进行行列排序，排序的原则是根据x坐标的从大到小的变化，如果发生变化则判定为新一行的开始
        for j in range(len(position[0]) - 1):
            # print(j)
            if position[0][j] - position[0][j + 1] > 0:
                ques_num.append(i)
                i = 0
            else:
                i += 1
        ques_num.append(i)
        # 例如结果ques_num[2,2,1] ，表示第一行3题，第二行3题，第三行2题

        num = 0
        i = 0  # 代表目前处理的题号
        # 将宽高同时大于5px,则symbol为候选对象，如果小于5px,则symbol作为杂质祛除
        for j in range(len(self.boxes_answer)):
            if self.boxes_answer[j][1]['w'] > 5 and self.boxes_answer[j][1]['h'] > 5:
                space.append(j)

        for t in range(len(ques_num)):
            if t == 0:
                num += ques_num[t]
            else:
                num += ques_num[t] + 1
            while i <= num:
                # print("第", i, "题")
                # print(ques_num)
                # for r in remove_item:
                #     space.remove(r)
                remove_item.clear()
                y2List.clear()
                y1List.clear()
                x2List.clear()
                x1List.clear()
                dict.clear()
                temp2.clear()
                if t == 0:  # 找最顶层中答案区
                    for j in space:
                        if detectUpLeft(self.boxes_answer, position, Q_position, i, j, ques_num[t]):
                            x1List.append(self.boxes_answer[j][1]['x'] + Q_position[0])
                            y1List.append(self.boxes_answer[j][1]['y'] + Q_position[1])
                            y2List.append(self.boxes_answer[j][1]['y'] + self.boxes_answer[j][1]['h'] + Q_position[1])
                            if (self.boxes_answer[j][1]['x'] + self.boxes_answer[j][1]['w'] + Q_position[0] >
                                    position[0][i + 1]):
                                x2List.append(position[0][i + 1])
                            elif (self.boxes_answer[j][1]['x'] + self.boxes_answer[j][1]['w'] + Q_position[0]
                                  < position[0][i + 1]):
                                x2List.append(
                                    self.boxes_answer[j][1]['x'] + self.boxes_answer[j][1]['w'] + Q_position[0])
                                remove_item.append(j)
                        elif i == num:
                            if detectUpRight(self.boxes_answer, position, Q_position, i, j, ques_num[t]):
                                x1List.append(self.boxes_answer[j][1]['x'] + Q_position[0])
                                y1List.append(self.boxes_answer[j][1]['y'] + Q_position[1])
                                y2List.append(
                                    self.boxes_answer[j][1]['y'] + self.boxes_answer[j][1]['h'] + Q_position[1])
                                x2List.append(
                                    self.boxes_answer[j][1]['x'] + self.boxes_answer[j][1]['w'] + Q_position[0])
                                remove_item.append(j)
                    dict = cropAns(x1List, y1List, x2List, y2List, Q_position, position, i, num, ques_num[t],
                                   'up',
                                   temp1)
                    temp2.append(dict)
                    temp3 = copy.deepcopy(temp2)
                    temp1.append(temp3)
                    # print(space)
                elif t == len(ques_num) - 1:  # 找最底层中答案区
                    for j in space:
                        if i == num:
                            if detectLowRight(self.boxes_answer, position, Q_position, i, j, ques_num[t]):
                                x1List.append(self.boxes_answer[j][1]['x'] + Q_position[0])
                                y1List.append(self.boxes_answer[j][1]['y'] + Q_position[1])
                                y2List.append(
                                    self.boxes_answer[j][1]['y'] + self.boxes_answer[j][1]['h'] + Q_position[1])
                                x2List.append(
                                    self.boxes_answer[j][1]['x'] + self.boxes_answer[j][1]['w'] + Q_position[0])
                                remove_item.append(j)
                        elif detectLowLeft(self.boxes_answer, position, Q_position, i, j, ques_num[t]):
                            x1List.append(self.boxes_answer[j][1]['x'] + Q_position[0])
                            y1List.append(self.boxes_answer[j][1]['y'] + Q_position[1])
                            y2List.append(self.boxes_answer[j][1]['y'] + self.boxes_answer[j][1]['h'] + Q_position[1])
                            if self.boxes_answer[j][1]['x'] + self.boxes_answer[j][1]['w'] + Q_position[0] > \
                                    position[0][
                                        i + 1]:
                                x2List.append(position[0][i + 1])
                            elif (self.boxes_answer[j][1]['x'] + self.boxes_answer[j][1]['w'] + Q_position[0] <
                                  position[0][
                                      i + 1]):
                                x2List.append(
                                    self.boxes_answer[j][1]['x'] + self.boxes_answer[j][1]['w'] + Q_position[0])
                                remove_item.append(j)
                    dict = cropAns(x1List, y1List, x2List, y2List, Q_position, position, i, num, ques_num[t],
                                   'sub',
                                   temp1)
                    temp2.append(dict)
                    temp3 = copy.deepcopy(temp2)
                    temp1.append(temp3)
                    # print(space)
                else:  # 找中间层中答案区
                    for j in space:
                        if i == num:
                            if i + ques_num[t] + 1 >= len(position[0]):
                                if detectMiddleRightNoSub(self.boxes_answer, position, Q_position, i, j,
                                                          ques_num[t]):
                                    x1List.append(self.boxes_answer[j][1]['x'] + Q_position[0])
                                    y1List.append(self.boxes_answer[j][1]['y'] + Q_position[1])
                                    y2List.append(
                                        self.boxes_answer[j][1]['y'] + self.boxes_answer[j][1]['h'] + Q_position[1])
                                    x2List.append(
                                        self.boxes_answer[j][1]['x'] + self.boxes_answer[j][1]['w'] + Q_position[0])
                                    remove_item.append(j)
                            elif i + ques_num[t] + 1 < len(position[0]):
                                if detectMiddleRightSub(self.boxes_answer, position, Q_position, i, j, ques_num[t]):
                                    if self.boxes_answer[j][1]['y'] + Q_position[1] < temp1[i - ques_num[t] - 1][0][
                                        'y2']:
                                        y1List.append(temp1[i - ques_num[t] - 1][0]['y2'])
                                    else:
                                        y1List.append(self.boxes_answer[j][1]['y'] + Q_position[1])
                                    x1List.append(self.boxes_answer[j][1]['x'] + Q_position[0])
                                    y2List.append(
                                        self.boxes_answer[j][1]['y'] + self.boxes_answer[j][1]['h'] + Q_position[1])
                                    x2List.append(
                                        self.boxes_answer[j][1]['x'] + self.boxes_answer[j][1]['w'] + Q_position[0])
                                    remove_item.append(j)
                        elif i + ques_num[t] + 1 >= len(position[0]):
                            if detectMiddleLeftNoSub(self.boxes_answer, position, Q_position, i, j, ques_num[t]):
                                x1List.append(self.boxes_answer[j][1]['x'] + Q_position[0])
                                y1List.append(self.boxes_answer[j][1]['y'] + Q_position[1])
                                y2List.append(
                                    self.boxes_answer[j][1]['y'] + self.boxes_answer[j][1]['h'] + Q_position[1])
                                if (self.boxes_answer[j][1]['x'] + self.boxes_answer[j][1]['w'] + Q_position[0] >
                                        position[0][
                                            i + 1]):
                                    x2List.append(position[0][i + 1])
                                elif (self.boxes_answer[j][1]['x'] + self.boxes_answer[j][1]['w'] + Q_position[0] <
                                      position[0][
                                          i + 1]):
                                    x2List.append(
                                        self.boxes_answer[j][1]['x'] + self.boxes_answer[j][1]['w'] + Q_position[0])
                                    remove_item.append(j)
                        elif i + ques_num[t] + 1 < len(position[0]):
                            if detectMiddleLeftSub(self.boxes_answer, position, Q_position, i, j, ques_num[t]):
                                x1List.append(self.boxes_answer[j][1]['x'] + Q_position[0])
                                y1List.append(self.boxes_answer[j][1]['y'] + Q_position[1])
                                y2List.append(
                                    self.boxes_answer[j][1]['y'] + self.boxes_answer[j][1]['h'] + Q_position[1])
                                if (self.boxes_answer[j][1]['x'] + self.boxes_answer[j][1]['w'] + Q_position[0] >
                                        position[0][
                                            i + 1]):
                                    x2List.append(position[0][i + 1])
                                elif (self.boxes_answer[j][1]['x'] + self.boxes_answer[j][1]['w'] + Q_position[0] <
                                      position[0][
                                          i + 1]):
                                    x2List.append(
                                        self.boxes_answer[j][1]['x'] + self.boxes_answer[j][1]['w'] + Q_position[0])
                                    remove_item.append(j)
                    dict = cropAns(x1List, y1List, x2List, y2List, Q_position, position, i, num, ques_num[t],
                                   'middle',
                                   temp1)
                    temp2.append(dict)
                    temp3 = copy.deepcopy(temp2)
                    temp1.append(temp3)
                    # print(space)
                i += 1
                index.append(i)
        position_with_answer = [([0] * len(position[0])) for i in range(5)]
        for i in range(len(position[0])):
            ansPos = preUseCropAns(temp1[i][0]['x1'], temp1[i][0]['y1'], temp1[i][0]['x2'], temp1[i][0]['y2'])
            oralBox = [position[0][i], position[1][i], position[2][i], position[3][i]]
            ansPos = self.cropAns(ansPos, oralBox, i)
            position_with_answer[0][i] = position[0][i]  # 题干左上角x坐标
            position_with_answer[1][i] = ansPos[0]  # 答案左上角坐标x坐标
            position_with_answer[2][i] = ansPos[1]  # 答案左上角坐标y坐标
            position_with_answer[3][i] = ansPos[2]  # 答案右下角坐标x坐标
            position_with_answer[4][i] = ansPos[3]  # 答案右下角坐标y坐标
        return position_with_answer


if __name__ == '__main__':

    imname = 'D://exam_img/real_image/full/1-1.jpg'  # 母卷路径
    imagePath = 'D://exam_img/answer_image/full/1-1.jpg'  # 答案卷路径
    '''
    position[[x0,x0,...],[y0,y0,....],[x1,x1,....],[y1,y1,....]]
    position已进行排序（从左到右，从上到下）
    # x0：口算题题干左上角x坐标
    # y0: 口算题题干左上角y坐标
    # x1：口算题题干右下角x坐标
    # y1: 口算题题干右下角y坐标
    '''
    position = [
        [330, 885, 1441, 331, 886, 1443, 331, 886, 1442, 332, 886, 1442, 332, 888, 1443, 332, 887, 1443, 333, 888],
        [997, 995, 992, 1090, 1088, 1085, 1184, 1182, 1179, 1277, 1274, 1272, 1371, 1369, 1368, 1465, 1463, 1461, 1558,
         1555],
        [543, 1098, 1653, 516, 1099, 1654, 516, 1099, 1654, 572, 1100, 1655, 545, 1100, 1655, 546, 1171, 1754, 616,
         1200],
        [1036, 1033, 1031, 1130, 1127, 1125, 1224, 1221, 1219, 1316, 1317, 1316, 1410, 1408, 1406, 1504, 1502, 1500,
         1598, 1596]]
    # 口算题大题区域坐标
    Q_position = [228, 949, 2182, 1646]
    ans_img = cv2.imread(imagePath)
    cvModelImg = cv2.imread(imname)
    # AnsImg = preprocess.align(imname, imagePath)
    AnsImg = preprocess.separateAns(imname, imagePath)
    calAnsPos = calAnsPos(Q_position, AnsImg, cvModelImg)
    position_with_answer = calAnsPos.answer_zone(position, Q_position)
    for i in range(len(position_with_answer[0])):
        cv2.rectangle(AnsImg, (position_with_answer[1][i], position_with_answer[2][i]),
                      (position_with_answer[3][i], position_with_answer[4][i]), 0, 3)
    cv2.namedWindow(imagePath, cv2.WINDOW_NORMAL)
    cv2.imshow(imagePath, AnsImg)
    cv2.waitKey()
    cv2.destroyAllWindows()

    '''
     imname = 'D://exam_img/real_image/tianzheng/real/jieduanceshi/tz-1-model-1.jpg'
    imageAnsDir = 'D://exam_img/answer_image/tianzheng/ans/jieduan/1/'
    position = [
        [535, 1299, 1722, 530, 921, 1293, 1693, 530, 921, 1292, 1691, 532, 922, 1297, 1691, 532, 923, 1294, 1670],
        [721, 714, 712, 812, 809, 806, 804, 905, 901, 899, 896, 997, 994, 990, 988, 1089, 1086, 1083, 1081],
        [667, 1500, 1894, 729, 1146, 1448, 1888, 661, 1101, 1472, 1866, 708, 1170, 1543, 1913, 778, 1194, 1496, 1983],
        [764, 757, 755, 855, 852, 849, 847, 948, 944, 942, 939, 1040, 1037, 1033, 1031, 1132, 1129, 1126, 1124]]

    Q_position = [0, 692, 2303, 1264]
    cvModelImg = cv2.imread(imname)
    pathDir = os.listdir(imageAnsDir)
    for t in pathDir:
        print("正在处理：", t)
        imagePath_answer = imageAnsDir + t
        # cv_img2 = preprocess.align(imname, imagePath_answer)
        AnsImg = preprocess.separateAns(imname, imagePath_answer)
        calAnsPosT = calAnsPos(Q_position, AnsImg, cvModelImg)
        position_with_answer = calAnsPosT.answer_zone(position, Q_position)
        for i in range(len(position_with_answer[0])):
            cv2.rectangle(AnsImg, (position_with_answer[1][i], position_with_answer[2][i]),
                          (position_with_answer[3][i], position_with_answer[4][i]), 0, 3)
        cv2.imwrite('D://exam_img/result/oralAnsResult/20190628/tianzheng/jieduan/01/' + str(t), AnsImg)
        del calAnsPosT
    '''
