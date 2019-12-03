# 作者：简卓林
# 日期：2019.7.12
# Version:1.0.0
import json
import cv2
import numpy as np
# from tesserocr import PyTessBaseAPI, RIL, Image
import duo_template_tianzheng, handleTextLine
import preprocess
import crop_answer
import ocrjietu2
import imutils
import locale
from contextlib import contextmanager


class gapFillAnsPos(object):
    """
        填空题答案区选出类
    """

    @contextmanager
    def c_locale(self):
        """
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

    def __init__(self, sepAnsImg, alignAnsImg, modelImage, content):
        """
                 __init__(self, sepAnsImg, alignAnsImg, modelImage, content):初始化数据为了给后面的答案区选出函数使用
                :param sepAnsImg: 分离卷
                :param alignAnsImg: 对齐卷
                :param modelImage: 母卷
                :param content: 答案库内容
                初始化数据内容为：
                    content:答案库内容
                    sepAnsImg:分离后的答案卷（numpy格式)
                    alignAnsImg:对齐后的答案卷(numpy格式)
                    handledTextLine:母卷(双栏)每一行的行高
                    sepAns:通过tesseract扫描分离卷后获取的答案卷的内容(symbol)
                    aliAns:通过tesseract扫描对其卷后获取内容(symbol)
        """

        # 扫描答案卷
        def ans_scan(ansImage, levelUsage):
            """
                ans_scan(ansImage, levelUsage):tesseract扫描图片函数
                :param ansImage:答案卷
                :param levelUsage:tesseract的level
                :return:基于symbol的box{x,y,w,h}
            """
            pil_im = Image.fromarray(ansImage)
            with PyTessBaseAPI(psm=6, lang="chi_sim") as api:
                api.SetImage(pil_im)
                boxes = api.GetComponentImages(levelUsage, True)
            return boxes

        self.content = content
        self.sepAnsImg = sepAnsImg
        self.alignAnsImg = cv2.cvtColor(alignAnsImg, cv2.COLOR_BGR2GRAY)
        self.handledTextLine = handleTextLine.getHandleTextLine(modelImage)
        with self.c_locale():
            from tesserocr import RIL, PyTessBaseAPI, Image

            self.sepAns = ans_scan(sepAnsImg, RIL.SYMBOL)
            self.aliAns = ans_scan(alignAnsImg, RIL.SYMBOL)

    def cropAns(self, ansPostion, fillBox, i):
        """
        cropAns(self, ansPostion, fillBox, i):用于裁剪白边
        :param ansPostion:基于答案区选出算法后得到的box[x0,y0,x1,y1]
        :param fillBox:基于母卷扫描后得到圆圈方框括号的box[x0,y0,x1,y1,unknown,type]
        :param i:fillBoxes所属的fillBox所在的index(只用于测试使用)
        :return:裁取白边后得到的新答案box[x0,y0,x1,y1]
        """

        def searchWhichTextline(fillBox, handledTextLine):
            """
            searchWhichTextline(fillBox, handledTextLine):获取答案所在行的行高
            :param fillBox: 基于母卷扫描后的到圆圈方框括号的box[x0,y0,x1,y1,unknown,type]
            :param handledTextLine:母卷(双栏)每一行的行高
            :return:答案区所在的行区域item{x,y,w,h}
            """
            # print("handleAll", handledTextLine)
            for item in handledTextLine:
                # print(item)
                if item['x'] <= fillBox[0] <= fillBox[2] <= item['x'] + item['w'] and item['y'] <= fillBox[1] <= \
                        fillBox[3] <= item['y'] + item['h']:
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

        unbw = self.sepAnsImg[ansPostion[1]:ansPostion[3], ansPostion[0]:ansPostion[2]]
        handledTextLine = searchWhichTextline(fillBox, self.handledTextLine)
        # print("handledTextLine", handledTextLine)
        if handledTextLine:
            # print('i:', i)
            if ansPostion[1] < handledTextLine['y']:
                ansPostion[1] = handledTextLine['y']
                bw = 255 - self.sepAnsImg[ansPostion[1]:ansPostion[3], ansPostion[0]:ansPostion[2]]
                try:
                    topSub = topImpurityCut(bw)
                    # print('topSub:', topSub)
                    ansPostion[1] = ansPostion[1] + topSub
                except:
                    ansPostion[1] = ansPostion[1]
            if ansPostion[3] > handledTextLine['y'] + handledTextLine['h']:
                ansPostion[3] = handledTextLine['y'] + handledTextLine['h']
                bw = 255 - self.sepAnsImg[ansPostion[1]:ansPostion[3], ansPostion[0]:ansPostion[2]]
                try:
                    bottomSub = bottomImpurityCut(bw)
                    # print('bottomSub:', bottomSub)
                    ansPostion[3] = ansPostion[3] - bottomSub
                except:
                    ansPostion[3] = ansPostion[3]
        bw = 255 - self.sepAnsImg[ansPostion[1]:ansPostion[3], ansPostion[0]:ansPostion[2]]
        # if i == 46:
        #     cv2.imwrite('template/test.jpg', bw)
        #     cv2.imwrite('template/unbw.jpg',unbw)
        try:
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

    # 扫描填空题
    def AnsPos(self, midPos, fillBoxes, datiNum, midTitle):
        """
        AnsPos(self, midPos, fillBoxes, datiNum, midTitle):通过答案区选出算法获取未裁取白边处理的答案区
        :param midPos:中题的坐标[x0,y0,x1,y1]，如果中题在两面例[x0,y0,x1,y1,x2,y2,x3,y3]
        :param fillBoxes:基于tesseract扫描母卷的圆圈，括号，方框的坐标<numpy>
        :param datiNum:答案库大题的index
        :param midTitle:答案库中题的index
        :return:每道中题答案区的坐标集ansPositionFill<numpy>
        """

        # 检测括号的答案区
        def detectInBracket(boxes, b, j, i):
            """
           detectInBracket(boxes, b, j, i):检测tesseract扫描出来的symbol是否在括号的区域内
           :param boxes: 基于tesseract扫描答案卷的得到的boxes[{x,y,w,h}]
           :param b: 基于母卷获取的括号坐标[x0,y0,x1,y1]
           :param j: box所属的boxes的index
           :param i: 基于母卷获取的圆圈、括号、方框坐标的index
           :return:0：不在括号内  1:在括号内
           """
            bottom = boxes[j][1]['y'] + boxes[j][1]['h']
            up = boxes[j][1]['y']
            upLine = b[i][1]
            middleLine = b[i][1] + (b[i][3] - b[i][1]) / 2
            bottomLine = b[i][3]
            if b[i][0] - 15 < boxes[j][1]['x'] < b[i][2] + 15:
                if (bottom < b[i][3] + 0.5 * (b[i][3] - b[i][1])
                        and up > b[i][1] - 0.5 * (b[i][3] - b[i][1])):
                    return 1
                elif up <= middleLine <= bottom:
                    return 1
                elif up <= upLine <= bottom:
                    return 1
                elif up < bottomLine <= bottom:
                    return 1
            else:
                return 0

        # 检测圆/方的答案区
        def detectInCirSqr(boxes, cs, j, i):
            """
            detectInCirSqr(boxes, cs, j, i):检测tesseract扫描出来的symbol是否在圆圈/方框的区域内
            :param boxes:  基于tesseract扫描答案卷的得到的boxes[{x,y,w,h}]
            :param cs: 基于母卷获取的圆圈/方框的坐标[x0,y0,x1,y1]
            :param j: box所属的boxes的index
            :param i: 基于母卷获取的圆圈、括号、方框坐标的index
            :return: 0：不在圆圈/方框内  1:在圆圈/方框内
            """
            if (cs[i][0] - 10 < boxes[j][1]['x'] < cs[i][2] + 10
                    and boxes[j][1]['y'] + boxes[j][1]['h'] < cs[i][3] + 10
                    and boxes[j][1]['y'] > cs[i][1] - 10):
                return 1
            else:
                return 0

        # 检测直线答案区
        def detectOnLine(boxes, l, j, i):
            """
            detectInCirSqr(boxes, cs, j, i):检测tesseract扫描出来的symbol是否在直线的区域内
            :param boxes: 基于tesseract扫描答案卷的得到的boxes[{x,y,w,h}]
            :param l:  基于母卷获取的直线的坐标[x0,y0,x1,y1]
            :param j: box所属的boxes的index
            :param i: 基于母卷获取的圆圈、括号、方框坐标的index
            :return: 0：不在直线区域内  1:在直线区域内
            """
            up = boxes[j][1]['y']
            bottom = boxes[j][1]['y'] + boxes[j][1]['h']
            upLine = l[i][1] - 50
            midLine = l[i][1] - 25
            downLine = l[i][1]
            if l[i][0] - 20 < boxes[j][1]['x'] < l[i][2] + (l[i][2] - l[i][0]) / 4:
                if up <= upLine <= bottom:
                    return 1
                elif up <= midLine <= bottom:
                    return 1
                elif up < downLine <= bottom:
                    return 1
                else:
                    return 0

        # 截取圆/方的答案区
        def ansPositionCirSqr(tempx1, tempy1, tempx2, tempy2, cs, alignAnsImg, sepAnsImg):
            """
            ansPositionCirSqr(tempx1, tempy1, tempx2, tempy2, cs, alignAnsImg, sepAnsImg):汇总圆圈/方框中symbol，得到答案区的左边界，有边界，上边界，下边界
            :param tempx1:答案区内含有的symbol的x0汇总后的列表
            :param tempy1:答案区内含有的symbol的y0汇总后的列表
            :param tempx2:答案区内含有的symbol的x1汇总后的列表
            :param tempy2:答案区内含有的symbol的y1汇总后的列表
            :param cs:基于母卷获取的圆圈/方框的坐标[x0,y0,x1,y1]
            :param alignAnsImg:对齐之后的答案卷
            :param sepAnsImg:分离之后的答案卷
            :return:答案区选出的坐标（未裁剪）
            """
            ansPositionCS = [0 for x in range(0, 4)]
            # print("ans_postionCS:", ansPositionCS)

            try:
                ansPositionCS[0] = min(tempx1)
                ansPositionCS[1] = min(tempy1)
                ansPositionCS[2] = max(tempx2)
                ansPositionCS[3] = max(tempy2)
            except Exception:
                if ansPositionCS == [0, 0, 0, 0]:
                    # print(cs[:4], cs[5])
                    rect, is_empty = crop_answer.giveme_answer(alignAnsImg, sepAnsImg, cs[:4], cs[5])
                    ansPositionCS = rect
                else:
                    ansPositionCS[0] = cs[0]
                    ansPositionCS[1] = cs[1]
                    ansPositionCS[2] = cs[2]
                    ansPositionCS[3] = cs[3]

            return ansPositionCS

        # 截取括号的答案区
        def ansPositionBracket(tempx1, tempy1, tempx2, tempy2, b, alignAnsImg, sepAnsImg):
            """
            ansPositionBracket(tempx1, tempy1, tempx2, tempy2, b, alignAnsImg, sepAnsImg):汇总括号中的symbol，得到答案区的左边界，有边界，上边界，下边界
            :param tempx1:答案区内含有的symbol的x0汇总后的列表
            :param tempy1:答案区内含有的symbol的y0汇总后的列表
            :param tempx2:答案区内含有的symbol的x1汇总后的列表
            :param tempy2:答案区内含有的symbol的y1汇总后的列表
            :param b:基于母卷获取的方框的坐标[x0,y0,x1,y1]
            :param alignAnsImg:对齐之后的答案卷
            :param sepAnsImg:分离之后的答案卷
            :return:答案区选出的坐标（未裁剪）
            """
            # print("tempx1", tempx1)
            # print("tempy1", tempy1)
            # print("tempx2", tempx2)
            # print("tempy2", tempy2)
            ansPositionB = [0 for x in range(0, 4)]
            # print("ans_postionB:", ansPositionB)

            try:
                ansPositionB[0] = min(tempx1)
                ansPositionB[1] = min(tempy1)
                ansPositionB[2] = max(tempx2)
                ansPositionB[3] = max(tempy2)
                exp = []
            except Exception:
                if ansPositionB == [0, 0, 0, 0]:
                    rect, is_empty = crop_answer.giveme_answer(alignAnsImg, sepAnsImg, b[:4], b[5])
                    ansPositionB = list(rect)
                else:
                    ansPositionB[0] = b[0]
                    ansPositionB[1] = b[1] - 0.1 * (b[3] - b[1])
                    ansPositionB[2] = b[2]
                    ansPositionB[3] = b[3] + 0.1 * (b[3] - b[1])
                exp = ansPositionB.copy()
            return ansPositionB, exp

        # 截取直线的答案区
        def ansPositionLine(tempx1, tempy1, tempx2, tempy2, l):
            """
            ansPositionLine(tempx1, tempy1, tempx2, tempy2, l):汇总直线区域中的symbol，得到答案区的左边界，有边界，上边界，下边界
            :param tempx1:答案区内含有的symbol的x0汇总后的列表
            :param tempy1:答案区内含有的symbol的y0汇总后的列表
            :param tempx2:答案区内含有的symbol的x1汇总后的列表
            :param tempy2:答案区内含有的symbol的y1汇总后的列表
            :param l: 基于母卷获取的直线的坐标[x0,y0,x1,y1]
            :return: 答案区选出的坐标（未裁剪）
            """
            ansPositionL = [0 for x in range(0, 4)]
            # print("ans_postionL:", ansPositionL)

            try:
                ansPositionL[0] = min(tempx1)
                ansPositionL[1] = min(tempy1)
                ansPositionL[2] = max(tempx2)
                ansPositionL[3] = max(tempy2)
            except Exception:
                ansPositionL[0] = l[0]
                ansPositionL[1] = l[1] - 100
                ansPositionL[2] = l[2]
                ansPositionL[3] = l[3]
            return ansPositionL

        # 排序
        def order(midPos, fillBoxes):
            """
            order(midPos, fillBoxes):对中题里的答案进行答案进行排序
            :param midPos:中题的坐标[x0,y0,x1,y1]
            :param fillBoxes:基于tesseract扫描母卷的圆圈，括号，方框的坐标<numpy>
            :return:quesNum:答案的数量 items:排序答案区<numpy>
            """

            # 快速排序
            def quickSort(data, start, end):
                """
                quickSort(data, start, end):快速排序列表
                :param data: 需要排序的数据
                :param start: 起始index
                :param end: 结束index
                :return: 排序完之后的列表
                """
                i = start
                j = end
                # i与j重合时，一次排序结束
                if i >= j:
                    return
                # 设置最左边的数为基准值
                flag = data[start]
                while i < j:
                    while i < j and data[j] >= flag:
                        j -= 1
                    # 找到右边第一个小于基准的数，赋值给左边i。此时左边i被记录在flag中
                    data[i] = data[j]
                    while i < j and data[i] <= flag:
                        i += 1
                    # 找到左边第一个大于基准的数，赋值给右边的j。右边的j的值和上面左边的i的值相同
                    data[j] = data[i]
                # 由于循环以i结尾，循环完毕后把flag值放到i所在位置。
                data[i] = flag
                # 除去i之外两段递归
                quickSort(data, start, i - 1)
                quickSort(data, i + 1, end)

            index = []
            y1 = []
            numInRow = 0
            tempDel = []
            quesNum = []
            x1Index = {}
            keys = []
            values = []
            items = []
            x1 = []
            for i in range(len(fillBoxes)):
                if (midPos[3] > fillBoxes[i][1] + (fillBoxes[i][3] - fillBoxes[i][1]) / 2 > midPos[1] and midPos[0] <
                        fillBoxes[i][0] < midPos[2]):
                    index.append(i)
                    if fillBoxes[i][5] == 2:
                        y1.append(fillBoxes[i][1] - 15)
                    else:
                        y1.append(fillBoxes[i][1])
                else:
                    y1.append(float("inf"))
            # print("index:", index)
            while index:
                tempDel.clear()
                x1Index.clear()
                x1.clear()
                keys.clear()
                values.clear()
                for item in index:
                    if min(y1) <= fillBoxes[item][1] <= min(y1) + 30:
                        tempDel.append(item)
                        numInRow += 1
                    # if fillBoxes[item][5] == 0 or fillBoxes[item][5] == 1:
                    #     if min(y1) <= fillBoxes[item][1] <= min(y1) + 30:
                    #         tempDel.append(item)
                    #         numInRow += 1
                    # elif fillBoxes[item][5] == 2:
                    #     if min(y1) <= fillBoxes[item][1] <= min(y1) + 30:
                    #         tempDel.append(item)
                    #         numInRow += 1
                # print("tempDel:", tempDel)
                quesNum.append(numInRow)
                numInRow = 0
                for d in tempDel:
                    x1Index[d] = fillBoxes[d][0]
                    x1.append(fillBoxes[d][0])
                    index.remove(d)
                    y1[d] = float("inf")
                quickSort(x1, 0, len(x1) - 1)
                keys = list(x1Index.keys())
                # print("keys:", keys)
                values = list(x1Index.values())
                for i in x1:
                    items.append(keys[values.index(i)])
                    keys.remove(keys[values.index(i)])
                    values.remove(i)
                # print("delete after:", keys)
            # print("items:", items)
            return quesNum, items

        # 逆序方案
        def crop(ansPositionFill, datiNum, midTitle, content=self.content):
            """
            crop(ansPositionFill, datiNum, midTitle, content=self.content):用于删除中题题干的伪答案区
            :param ansPositionFill: 每道中题答案区的坐标集ansPositionFill
            :param datiNum: 答案库大题的index
            :param midTitle: 答案库中题的index
            :param content: 答案库内容
            :return: 裁剪
            """
            nowNum = len(ansPositionFill)
            correctNum = len(content["answerList"][datiNum]["paperQuestionList"][midTitle]["answer"])
            # print('扫描出的答案个数：', nowNum)
            # print("答案数目：", correctNum)
            if content["answerList"][datiNum]["paperQuestionList"][midTitle]['autoMarkFlag'] == 1:
                if nowNum >= correctNum:
                    ansPositionFill = ansPositionFill.tolist()[nowNum - correctNum:]
                    # print(ansPositionFill)
                else:
                    ansPositionFill = ansPositionFill.tolist()
            else:
                ansPositionFill = []
            # print("裁剪结果：", ansPositionFill)
            return ansPositionFill

        tempx1 = []
        tempx2 = []
        tempy1 = []
        tempy2 = []
        ansPositionFill = np.empty([0, 4], dtype=np.int32)
        fillBoxes = np.array(fillBoxes).astype(np.int)
        r = 0
        while r < len(midPos) / 4:
            quesNum, indexCopy = order(midPos[r * 4:(r + 1) * 4], fillBoxes)
            # print(order(midPos, fillBoxes))
            midTitlePosFill = np.empty([len(indexCopy), 4], dtype=np.int32)
            t = 0
            for i in indexCopy:
                tempx1.clear()
                tempx2.clear()
                tempy1.clear()
                tempy2.clear()

                if fillBoxes[i][5] == 1:
                    for j in range(len(self.aliAns)):
                        if detectInBracket(self.aliAns, fillBoxes, j, i):
                            if self.aliAns[j][1]['x'] + self.aliAns[j][1]['w'] + 5 < fillBoxes[i][2]:
                                tempx2.append(self.aliAns[j][1]['x'] + self.aliAns[j][1]['w'])
                            elif self.aliAns[j][1]['x'] + 20 < fillBoxes[i][2]:
                                tempx2.append(fillBoxes[i][2] - 10)
                            elif not tempx2:
                                tempx2.append(fillBoxes[i][2])
                            tempx1.append(self.aliAns[j][1]['x'])
                            tempy1.append(self.aliAns[j][1]['y'])
                            tempy2.append(self.aliAns[j][1]['y'] + self.aliAns[j][1]['h'])
                    aliAfterPos, exp1 = ansPositionBracket(tempx1, tempy1, tempx2, tempy2, fillBoxes[i],
                                                           self.alignAnsImg,
                                                           self.sepAnsImg)
                    tempx1.clear()
                    tempx2.clear()
                    tempy1.clear()
                    tempy2.clear()
                    for j in range(len(self.sepAns)):
                        if detectInBracket(self.sepAns, fillBoxes, j, i):
                            tempx2.append(self.sepAns[j][1]['x'] + self.sepAns[j][1]['w'])
                            tempx1.append(self.sepAns[j][1]['x'])
                            tempy1.append(self.sepAns[j][1]['y'])
                            tempy2.append(self.sepAns[j][1]['y'] + self.sepAns[j][1]['h'])
                    sepAfterPos, exp2 = ansPositionBracket(tempx1, tempy1, tempx2, tempy2, fillBoxes[i],
                                                           self.alignAnsImg,
                                                           self.sepAnsImg)
                    if len(exp2) > 0:
                        # print("切换试卷")
                        midTitlePosFill[t] = aliAfterPos.copy()
                    else:
                        midTitlePosFill[t] = sepAfterPos.copy()
                elif fillBoxes[i][5] == 0:
                    for j in range(len(self.sepAns)):
                        if detectInCirSqr(self.sepAns, fillBoxes, j, i):
                            if self.sepAns[j][1]['x'] + self.sepAns[j][1]['w'] + 5 < fillBoxes[i][2]:
                                tempx2.append(self.sepAns[j][1]['x'] + self.sepAns[j][1]['w'])
                            elif self.sepAns[j][1]['x'] + 20 < fillBoxes[i][2]:
                                tempx2.append(fillBoxes[i][2] - 10)
                            elif not tempx2:
                                tempx2.append(fillBoxes[i][2])
                            tempx1.append(self.sepAns[j][1]['x'])
                            tempy1.append(self.sepAns[j][1]['y'])
                            tempy2.append(self.sepAns[j][1]['y'] + self.sepAns[j][1]['h'])
                    midTitlePosFill[t] = ansPositionCirSqr(tempx1, tempy1, tempx2, tempy2, fillBoxes[i],
                                                           self.alignAnsImg,
                                                           self.sepAnsImg)
                elif fillBoxes[i][5] == 2:
                    for j in range(len(self.sepAns)):
                        if detectOnLine(self.sepAns, fillBoxes, j, i):
                            tempx2.append(self.sepAns[j][1]['x'] + self.sepAns[j][1]['w'])
                            tempx1.append(self.sepAns[j][1]['x'])
                            tempy1.append(self.sepAns[j][1]['y'])
                            tempy2.append(self.sepAns[j][1]['y'] + self.sepAns[j][1]['h'])
                    midTitlePosFill[t] = ansPositionLine(tempx1, tempy1, tempx2, tempy2, fillBoxes[i])
                midTitlePosFill[t] = self.cropAns(midTitlePosFill[t], fillBoxes[i], t)
                t += 1
            # print("midTitlePosFill:", midTitlePosFill)
            ansPositionFill = np.insert(ansPositionFill, len(ansPositionFill), midTitlePosFill, 0)
            r += 1
        # print('ansPositionFill:', ansPositionFill)
        ansPositionFill = crop(ansPositionFill, datiNum, midTitle)
        return ansPositionFill

    # 批处理
    def ansSelect(self, imname, data, datiNum):
        """
        ansSelect(self, imname, data, datiNum):该类的主函数，用于调用答案区选出函数及对大题实现答案区的整合
        :param imname: 母卷的路径
        :param data: 大题及中题的坐标<dict>
        :param datiNum: 大题index
                datiNum 为了匹配答案库的index,所以如果是第二面的答案库，它的index也是从0开始的
        :return: 答案区选出总的坐标<dict>
        """
        midIndex = []
        allAnsPosDict = {}
        '''
            调用多模板匹配的方法，得到母卷中填空题答案的坐标
        '''
        ocrjie = ocrjietu2.Jietemplate()
        ocrjie.ocrtemplate(imname)
        fill = duo_template_tianzheng.Template()
        fillBoxes = fill.totalmatch(imname)
        # print("大题Index：", datiNum)
        for midTitle in data:
            midIndex.append(midTitle)
        for i in range(len(midIndex)):
            # print("##############################################################################################")
            # print(i)
            midAnsPos = self.AnsPos(data[midIndex[i]], fillBoxes, datiNum, i)
            # print(midIndex[i], '题')
            allAnsPosDict[midIndex[i]] = midAnsPos

        return allAnsPosDict


if __name__ == '__main__':

    # 普通处理

    imname = 'D://exam_img/real_image/full/17-1.jpg'  # 母卷路径
    imagePath_answer = "D://exam_img/answer_image/full/17-1.jpg"  # 答案卷路径
    '''
       例如：data = [{4: {1: [2378, 1531, 4364, 2022], 2: [2378, 2022, 4364, 2935]}, 3: {1: [2378, 1053, 4364, 1444]},
                2: {1: [218, 1716, 2182, 2092], 2: [218, 2092, 2182, 2184], 3: [218, 2184, 2182, 2375],
                    4: [218, 2375, 2182, 2656], 5: [218, 2656, 2182, 2935], 6: [2378, 210, 4364, 535],
                    7: [2378, 535, 4364, 893], 8: [2378, 893, 4364, 1000]}, 1: {1: [218, 949, 2182, 1646]}},
               {7: {1: [2380, 2009, 4375, 2465], 2: [2380, 2465, 4375, 2936]},
                6: {1: [234, 2101, 2187, 2936], 2: [2380, 206, 4375, 606], 3: [2380, 606, 4375, 1001],
                    4: [2380, 1001, 4375, 1484], 5: [2380, 1484, 4375, 1943]},
                5: {1: [234, 544, 2187, 977], 2: [234, 977, 2187, 1497], 3: [234, 1497, 2187, 2034]},
                4: {3: [234, 196, 2187, 475]}}]
                第一行的key:4代表第四大题，里面的大括号代表里面中题的坐标及题号
       '''
    data = [{3: {1: [2372, 1642, 4358, 2934]},
             2: {1: [212, 1638, 2179, 2290], 2: [212, 2290, 2179, 2934], 3: [2372, 219, 4358, 617],
                 4: [2372, 617, 4358, 1001], 5: [2372, 1001, 4358, 1339], 6: [2372, 1339, 4358, 1457],
                 7: [2372, 1457, 4358, 1578]}, 1: {1: [212, 942, 2179, 1571]}},
            {6: {1: [2388, 2372, 4371, 2541], 2: [2388, 2541, 4371, 2934]},
             5: {1: [235, 1179, 2185, 1444], 2: [235, 1444, 2185, 1709], 3: [235, 1709, 2185, 2210],
                 4: [235, 2210, 2185, 2934, 2388, 0, 4371, 816], 5: [2388, 816, 4371, 2306]},
             4: {1: [235, 308, 2185, 569], 2: [235, 569, 2185, 829], 3: [235, 829, 2185, 1113]}}]

    anslibPath = "D://exam_img/anslib/35fen/17-12.json"  # 答案库的路径
    cv_img = preprocess.separateAns(imname, imagePath_answer)
    cv_img2 = preprocess.align(imname, imagePath_answer)
    cvModelImg = cv2.imread(imname)
    with open(anslibPath, 'r', encoding="UTF-8") as anslib:
        content = json.load(anslib)
        gapFillAnsPos = gapFillAnsPos(cv_img, cv_img2, cvModelImg, content)
        ans_position = gapFillAnsPos.ansSelect(imname, data[0][2], 1)
    # print("ans_position:", ans_position, type(ans_position[3]))
    for title in ans_position:
        for i in range(len(ans_position[title])):
            cv2.rectangle(cv_img, (ans_position[title][i][0], ans_position[title][i][1]),
                          (ans_position[title][i][2], ans_position[title][i][3]),
                          (0, 255, 0), 2)
            cv2.putText(cv_img, str(i), (ans_position[title][i][0], ans_position[title][i][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    # cv2.rectangle(cv_img,(234, 200),(2187, 299),(0,255,0),2)
    cv2.namedWindow(imagePath_answer, cv2.WINDOW_NORMAL)
    cv2.imshow(imagePath_answer, cv_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite("D://exam_img/result/gapFillAnsResult/20190524/order/2-1_new.jpg", cv_img)
