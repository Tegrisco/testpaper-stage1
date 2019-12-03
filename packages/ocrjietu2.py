import cv2 as cv
import numpy as np
import os
# from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageFilter
import locale
from contextlib import contextmanager
#from tesserocr import PyTessBaseAPI, PSM, iterate_level, RIL


class Jietemplate(object):
    def __init__(self):
        '''
            设置左括号判断条件列表self.l_pan及左括号坐标存储列表self.left_kuo
            设置右括号判断条件列表self.r_pan及右括号坐标存储列表self.right_kuo
            设置圆圈判断条件列表self.quan_pan及圆圈坐标存储列表self.shiyuanquan
            设置方块判断条件列表self.l_fangpan及方块坐标存储列表self.l_fangkuai
        '''
        self.width_hang = 0
        self.width_diao = 0

        self.left_kuo = []
        self.right_kuo = []
        self.shiyuanquan = []
        self.l_fangkuai = []
        self.r_fangkuai = []

        self.panduan = ['(', ')', '〇', '[', '口', 'L', '|', '“|', ':[', '[_]', '〔', '〉']
        self.l_pan = ['(']
        self.r_pan = [')']
        self.quan_pan = ['〇', '〔', '〉']
        self.l_fangpan = ['[', '口', 'L', '|', '“|', ':[', '[_]']
        self.r_fangpan = [']']

    @contextmanager
    def c_locale(self):
        try:
            currlocale = locale.getlocale()
        except ValueError:
            currlocale = ('en_US', 'UTF-8')
        # print ('Switching to C from {}'.format(currlocale))
        locale.setlocale(locale.LC_ALL, "C")
        yield
        # print ('Switching to {} from C'.format(currlocale))
        locale.setlocale(locale.LC_ALL, currlocale)

    def ocrtemplate(self, imname):
        '''
        1、输入图片，做灰度与二值化处理
        2、使用OCR按SYMBOL识别试卷图片
        3、对识别结果进行判断。若满足方块判断条件，则将坐标存放在方块列表中；
                若满足左括号判断条件，则将坐标存放在左括号列表中，以此类推.
        4、创建方块圆圈模板文件夹，左括号模板文件夹，右括号模板文件夹
        5、调用模板生成函数，生成对应的模板图片
        '''
        img = Image.open(imname)
        cvimg = cv.imread(imname)
        cvgray = cv.cvtColor(cvimg, cv.COLOR_BGR2GRAY)
        t_, cverzhi = cv.threshold(cvgray, 100, 255, cv.THRESH_BINARY_INV)
        im = img.filter(ImageFilter.SMOOTH_MORE)
        with self.c_locale():
            from tesserocr import PyTessBaseAPI, PSM, iterate_level, RIL
            with PyTessBaseAPI(psm=PSM.SINGLE_BLOCK, lang='chi_sim') as api:
                api.SetImage(im)
                api.SetVariable('save_blob_choices', 'T')
                api.Recognize()
                ri = api.GetIterator()
                level = RIL.SYMBOL
                for r in iterate_level(ri, level):
                    result = r.GetUTF8Text(level)
                    conf = r.Confidence(level)
                    coord = r.BoundingBox(level)

                    if result in self.panduan:
                        if (result in self.l_pan) and (coord[0] > 0) and (coord[2] > 0) and (conf > 95) and (
                                coord[2] - coord[0] + 12 < 35) and (coord[2] - coord[0] + 12 > 19) and (
                                coord[3] - coord[1] + 12 > 40):
                            self.left_kuo.append((coord[0] - 6, coord[1] - 6, coord[2] + 6, coord[3] + 6, conf,
                                                  coord[2] - coord[0] + 12, coord[3] - coord[1] + 12))

                        if (result in self.r_pan) and (coord[0] > 0) and (coord[2] > 0) and (conf > 95) and (
                                coord[2] - coord[0] + 12 < 35) and (coord[2] - coord[0] + 12 > 19) and (
                                coord[3] - coord[1] + 12 > 40):
                            self.right_kuo.append((coord[0] - 6, coord[1] - 6, coord[2] + 6, coord[3] + 6, conf,
                                                   coord[2] - coord[0] + 12, coord[3] - coord[1] + 12))

                        if (result in self.quan_pan) and (coord[0] > 0) and (coord[2] > 0) and (conf > 95) and (
                                coord[2] - coord[0] + 13 > 60) and (coord[2] - coord[0] + 13 < 120) and (
                                coord[3] - coord[1] + 10 > 60):
                            self.shiyuanquan.append((coord[0] - 5, coord[1] - 5, coord[2] + 8, coord[3] + 5, conf,
                                                     coord[2] - coord[0] + 13, coord[3] - coord[1] + 10))

                        if (result in self.l_fangpan) and (coord[0] > 0) and (coord[2] > 0) and (conf > 50) and (
                                coord[2] - coord[0] + 12 <= 100) and (coord[2] - coord[0] + 12 > 60) and (
                                coord[3] - coord[1] + 12 > 65):
                            self.l_fangkuai.append((coord[0] - 6, coord[1] - 6, coord[2] + 6, coord[3] + 6, conf,
                                                    coord[2] - coord[0] + 12, coord[3] - coord[1] + 12))
                    '''
                    if (result in self.r_fangkuai) and (coord[0]>0) and (coord[2]>0) and (
                            coord[2]-coord[0]<=100):# and (conf>90):
                        self.r_fangkuai.append((coord[0]-37,coord[1]-6,coord[2]+6,coord[3]+6,conf))
                    '''
                # if result:
                    # print(u'result:{},coord:{},conf:{}'.format(result, coord, conf))
        '''
        left_kuo=self.templatekuan(self.left_kuo)
        right_kuo=self.templatekuan(self.right_kuo)
        shiyuanquan=self.templatekuan(self.shiyuanquan)
        l_fangkuai=self.templatekuan(self.l_fangkuai)
        r_fangkuai=self.templatekuan(self.r_fangkuai)
        '''
        if not os.path.exists('template/template_tmp/fangyuan'):
            os.makedirs('template/template_tmp/fangyuan')
        if not os.path.exists('template/template_tmp/kuohao/leftkuo'):
            os.makedirs('template/template_tmp/kuohao/leftkuo')
        if not os.path.exists('template/template_tmp/kuohao/rightkuo'):
            os.makedirs('template/template_tmp/kuohao/rightkuo')

        self.lefttemplate(self.left_kuo, cvimg, cverzhi)
        self.righttemplate(self.right_kuo, cvimg, cverzhi)
        self.quantemplate(self.shiyuanquan, cvimg, cverzhi)
        self.fangtemplate(self.l_fangkuai, cvimg, cverzhi)

        # print('self.left_kuo:', np.array(self.left_kuo), '\n', len(self.left_kuo))
        # print('self.right_kuo:', np.array(self.right_kuo), '\n', len(self.right_kuo))
        # print('self.shiyuanquan:', np.array(self.shiyuanquan), '\n', len(self.shiyuanquan))
        # print('self.l_fangkuai', np.array(self.l_fangkuai), '\n', len(self.l_fangkuai))
        # print('self.r_fangkuai',r_fangkuai,'\n',len(self.r_fangkuai))

    def lefttemplate(self, left_kuo, cvimg, cverzhi):
        '''
        生成左括号模板图像，设置7个比例[1.3,1.3,1.4,1.25,0.9,0.7,0.9]
        将左括号坐标按高度从大到小排序，选取7个不同高度的左括号[h7,h6,h5,h4,h3,h2,h1]，
        并乘以比例因子[1.3*h7,1.3*h6,1.4*h5,1.25*h4,0.9*h3,0.7*h2,0.9*h1]得相应的值，
        用该值去调整左括的宽度
        在二值图中判断括号周围是否为空白区域，若为空白区域，在灰度图中截取左括号子图像作为模板

        '''
        left_kuo = np.array(left_kuo).astype(np.int16)
        if len(left_kuo):
            left_ind = np.argsort(left_kuo[:, 6])[::-1]
            left_kuo = left_kuo[left_ind]
            num = len(left_kuo)
            step = int(num / 2)
            #######################################
            for left in left_kuo:
                if left[6] < 60:
                    # print('left:', left)
                    k6 = int((left[6]) * 1.3)
                    erzhi6 = cverzhi[left[1]:left[3], left[2]:left[2] + k6]
                    # print('not np.sum(erzhi6)', not np.sum(erzhi6))
                    if not np.sum(erzhi6):
                        temp6 = cvimg[left[1]:left[3], left[0]:left[2] + k6, :]
                        # print('sddsafsadgsf:', temp6.shape)
                        # cv.imwrite('template/template_tmp/kuohao/leftkuo/temp6.jpg', temp6)
                        break
            ################################################
            '''
            if num>5:
                k6=int((left_kuo[0][6])*1.3)
                #print(left_kuo[0][1],left_kuo[0][3],left_kuo[0][0],left_kuo[0][2])
                erzhi6=cverzhi[left_kuo[0][1]:left_kuo[0][3],left_kuo[0][2]:left_kuo[0][2]+k6]
                print('not np.sum(erzhi6)',not np.sum(erzhi6))
                if not np.sum(erzhi6):
                    temp6=cvimg[left_kuo[0][1]:left_kuo[0][3],left_kuo[0][0]:left_kuo[0][2]+k6,:]
                    cv.imwrite('template/template_tmp/kuohao/leftkuo/temp6.jpg',temp6)
            '''
            if num >= 6:
                k6_1 = int((left_kuo[0 + 5][6]) * 1.3)
                # print(left_kuo[0+3][1],left_kuo[0+3][3],left_kuo[0+3][0],left_kuo[0+3][2])
                erzhi6_1 = cverzhi[left_kuo[0 + 5][1]:left_kuo[0 + 5][3], left_kuo[0 + 5][2]:left_kuo[0 + 5][2] + k6_1]
                if not np.sum(erzhi6_1):
                    temp6_1 = cvimg[left_kuo[5][1]:left_kuo[5][3], left_kuo[5][0]:left_kuo[5][2] + k6_1, :]
                    cv.imwrite('template/template_tmp/kuohao/leftkuo/temp6_1.jpg', temp6_1)

            if step - 2 >= 0:
                k5 = int(left_kuo[step - 2][6] * 1.4)
                erzhi5 = cverzhi[left_kuo[step - 2][1]:left_kuo[step - 2][3],
                         left_kuo[step - 2][2]:left_kuo[step - 2][2] + k5]
                # print('not np.sum(erzhi5)',not np.sum(erzhi5))
                if not np.sum(erzhi5):
                    temp5 = cvimg[left_kuo[step - 2][1]:left_kuo[step - 2][3],
                            left_kuo[step - 2][0]:left_kuo[step - 2][2] + k5, :]
                    cv.imwrite('template/template_tmp/kuohao/leftkuo/temp5.jpg', temp5)

            k4 = int(left_kuo[step][6] * 1.25)
            erzhi4 = cverzhi[left_kuo[step][1]:left_kuo[step][3], left_kuo[step][2]:left_kuo[step][2] + k4]
            # print('np.sum(erzhi4)',not np.sum(erzhi4))
            if not np.sum(erzhi4):
                temp4 = cvimg[left_kuo[step][1]:left_kuo[step][3], left_kuo[step][0]:left_kuo[step][2] + k4, :]
                cv.imwrite('template/template_tmp/kuohao/leftkuo/temp4.jpg', temp4)

            k2 = int(left_kuo[-1][6] * 0.7)
            erzhi2 = cverzhi[left_kuo[-1][1]:left_kuo[-1][3], left_kuo[-1][2]:left_kuo[-1][2] + k2]
            # print('np.sum(erzhi2)', not np.sum(erzhi2))
            if not np.sum(erzhi2):
                temp2 = cvimg[left_kuo[-1][1]:left_kuo[-1][3], left_kuo[-1][0]:left_kuo[-1][2] + k2, :]
                cv.imwrite('template/template_tmp/kuohao/leftkuo/temp2.jpg', temp2)

            if num >= 3:
                k3 = int(left_kuo[-3][6] * 0.9)
                erzhi3 = cverzhi[left_kuo[-3][1]:left_kuo[-3][3], left_kuo[-3][2]:left_kuo[-3][2] + k3]
                # print('np.sum(erzhi3)', not np.sum(erzhi3))
                if not np.sum(erzhi3):
                    temp3 = cvimg[left_kuo[-3][1]:left_kuo[-3][3], left_kuo[-3][0]:left_kuo[-3][2] + k3, :]
                    cv.imwrite('template/template_tmp/kuohao/leftkuo/temp3.jpg', temp3)

                k1 = int(left_kuo[-2][6] * 0.9)
                erzhi1 = cverzhi[left_kuo[-2][1]:left_kuo[-2][3], left_kuo[-2][2]:left_kuo[-2][2] + k1]
                # print('np.sum(erzhi3)', not np.sum(erzhi3))
                if not np.sum(erzhi1):
                    temp1 = cvimg[left_kuo[-2][1]:left_kuo[-2][3], left_kuo[-2][0]:left_kuo[-2][2] + k1, :]
                    cv.imwrite('template/template_tmp/kuohao/leftkuo/temp1.jpg', temp1)

    def righttemplate(self, right_kuo, cvimg, cverzhi):
        '''
        生成右括号模板图像，与左括号相似
        '''
        right_kuo = np.array(right_kuo).astype(np.int16)
        if len(right_kuo):
            right_ind = np.argsort(right_kuo[:, 6])[::-1]
            right_kuo = right_kuo[right_ind]
            num = len(right_kuo)
            step = int(num / 2)
            #########################################
            for right in right_kuo:
                if right[6] < 60:
                    k6 = int((right[6]) * 1.3)
                    erzhi6 = cverzhi[right[1]:right[3], right[0] - k6:right[0]]
                    # print('not np.sum(erzhi6)', not np.sum(erzhi6))
                    if not np.sum(erzhi6):
                        temp6 = cvimg[right[1]:right[3], right[0] - k6:right[2], :]
                        cv.imwrite('template/template_tmp/kuohao/rightkuo/temp6.jpg', temp6)
                        break
            #########################################
            '''
            if num>5:
                k6=int((right_kuo[0][6])*1.3)
                erzhi6=cverzhi[right_kuo[0][1]:right_kuo[0][3],right_kuo[0][0]-k6:right_kuo[0][0]]
                print('not np.sum(erzhi6)', not np.sum(erzhi6))
                if not np.sum(erzhi6):
                    temp6=cvimg[right_kuo[0][1]:right_kuo[0][3],right_kuo[0][0]-k6:right_kuo[0][2],:]
                    cv.imwrite('template/template_tmp/kuohao/rightkuo/temp6.jpg', temp6)
            '''
            if num >= 6:
                k6_1 = int((right_kuo[0 + 5][6]) * 1.3)
                # print(right_kuo[0+3][1],right_kuo[0+3][3],right_kuo[0+3][0],right_kuo[0+3][2])
                erzhi6_1 = cverzhi[right_kuo[0 + 5][1]:right_kuo[0 + 5][3],
                           right_kuo[0 + 5][0] - k6_1:right_kuo[0 + 5][0]]
                if not np.sum(erzhi6_1):
                    temp6_1 = cvimg[right_kuo[5][1]:right_kuo[5][3], right_kuo[5][0] - k6_1:right_kuo[5][2], :]
                    cv.imwrite('template/template_tmp/kuohao/rightkuo/temp6_1.jpg', temp6_1)

            if step - 2 >= 0:
                k5 = int(right_kuo[step - 2][6] * 1.4)
                erzhi5 = cverzhi[right_kuo[step - 2][1]:right_kuo[step - 2][3],
                         right_kuo[step - 2][0] - k5:right_kuo[step - 2][0]]
                # print('not np.sum(erzhi5)', not np.sum(erzhi5))
                if not np.sum(erzhi5):
                    temp5 = cvimg[right_kuo[step - 2][1]:right_kuo[step - 2][3],
                            right_kuo[step - 2][0] - k5:right_kuo[step - 2][2], :]
                    cv.imwrite('template/template_tmp/kuohao/rightkuo/temp5.jpg', temp5)

            k4 = int(right_kuo[step][6] * 1.25)
            erzhi4 = cverzhi[right_kuo[step][1]:right_kuo[step][3], right_kuo[step][0] - k4:right_kuo[step][0]]
            # print('np.sum(erzhi4)', not np.sum(erzhi4))
            if not np.sum(erzhi4):
                temp4 = cvimg[right_kuo[step][1]:right_kuo[step][3], right_kuo[step][0] - k4:right_kuo[step][2], :]
                cv.imwrite('template/template_tmp/kuohao/rightkuo/temp4.jpg', temp4)

            k2 = int(right_kuo[-1][6] * 0.7)
            erzhi2 = cverzhi[right_kuo[-1][1]:right_kuo[-1][3], right_kuo[-1][0] - k2:right_kuo[-1][0]]
            # print('np.sum(erzhi2)', not np.sum(erzhi2))
            if not np.sum(erzhi2):
                temp2 = cvimg[right_kuo[-1][1]:right_kuo[-1][3], right_kuo[-1][0] - k2:right_kuo[-1][2], :]
                cv.imwrite('template/template_tmp/kuohao/rightkuo/temp2.jpg', temp2)

            if num >= 3:
                k3 = int(right_kuo[-3][6] * 0.9)
                erzhi3 = cverzhi[right_kuo[-3][1]:right_kuo[-3][3], right_kuo[-3][0] - k3:right_kuo[-3][0]]
                # print('np.sum(erzhi3)',not np.sum(erzhi3))
                if not np.sum(erzhi3):
                    temp3 = cvimg[right_kuo[-3][1]:right_kuo[-3][3], right_kuo[-3][0] - k3:right_kuo[-3][2], :]
                    cv.imwrite('template/template_tmp/kuohao/rightkuo/temp3.jpg', temp3)

                k1 = int(right_kuo[-2][6] * 0.9)
                erzhi1 = cverzhi[right_kuo[-2][1]:right_kuo[-2][3], right_kuo[-2][0] - k1:right_kuo[-2][0]]
                # print('np.sum(erzhi3)',not np.sum(erzhi3))
                if not np.sum(erzhi1):
                    temp1 = cvimg[right_kuo[-2][1]:right_kuo[-2][3], right_kuo[-2][0] - k1:right_kuo[-2][2], :]
                    cv.imwrite('template/template_tmp/kuohao/rightkuo/temp1.jpg', temp1)

    def quantemplate(self, shiyuanquan, cvimg, cverzhi):
        '''
        将圆圈按宽度从大到小排列，选取3不同个宽度的圆圈坐标。
        在二值图中判断圆圈周围是否为空白区域，若为空白区域则从灰度图中截下该圆圈子图像作为模板；
                若为非空白区域，则设置循环50次，每次向右移动一个像素点，并判断周围是否为空白区域，直至截取圆圈子图像

        '''
        shiquan = np.array(shiyuanquan).astype(np.int16)
        if len(shiquan):
            shiquan_ind = np.argsort(shiquan[:, 5])[::-1]
            shiquan = shiquan[shiquan_ind]
            num = len(shiquan)
            step = int(num / 2)

            q3 = cverzhi[shiquan[0][1]:shiquan[0][3], shiquan[0][2]:shiquan[0][2] + 2]
            if not np.sum(q3):
                quan3 = cvimg[shiquan[0][1]:shiquan[0][3], shiquan[0][0]:shiquan[0][2], :]
                cv.imwrite('template/template_tmp/fangyuan/quan3.jpg', quan3)

            if step != 0:
                q2 = cverzhi[shiquan[step][1]:shiquan[step][3], shiquan[step][2]:shiquan[step][2] + 2]
                if not np.sum(q2):
                    quan2 = cvimg[shiquan[step][1]:shiquan[step][3], shiquan[step][0]:shiquan[step][2], :]
                    cv.imwrite('template/template_tmp/fangyuan/quan2.jpg', quan2)
                else:
                    q2_1 = cverzhi[shiquan[step][1]:shiquan[step][3], shiquan[step][2] + 13:shiquan[step][2] + 15]
                    if not np.sum(q2_1):
                        quan2_1 = cvimg[shiquan[step][1]:shiquan[step][3], shiquan[step][0]:shiquan[step][2] + 14, :]
                        cv.imwrite('template/template_tmp/fangyuan/quan2-1.jpg', quan2_1)
                    else:
                        for j in range(25):
                            q2_1_1 = cverzhi[shiquan[step][1]:shiquan[step][3],
                                     shiquan[step][2] + j:shiquan[step][2] + j + 1]
                            if not np.sum(q2_1_1):
                                quan2_1_1 = cvimg[shiquan[step][1]:shiquan[step][3],
                                            shiquan[step][0]:shiquan[step][2] + j + 5, :]
                                cv.imwrite('template/template_tmp/fangyuan/quan2-1_1.jpg', quan2_1_1)
                                break

            if num >= 3:
                q1 = cverzhi[shiquan[-1][1]:shiquan[-1][3], shiquan[-1][2]:shiquan[-1][2] + 2]
                if not np.sum(q1):
                    quan1 = cvimg[shiquan[-1][1]:shiquan[-1][3], shiquan[-1][0]:shiquan[-1][2] + 4, :]
                    cv.imwrite('template/template_tmp/fangyuan/quan1.jpg', quan1)
                else:
                    q1_1 = cverzhi[shiquan[-1][1]:shiquan[-1][3], shiquan[-1][2] + 13:shiquan[-1][2] + 15]
                    if not np.sum(q1_1):
                        quan1_1 = cvimg[shiquan[-1][1]:shiquan[-1][3], shiquan[-1][0]:shiquan[-1][2] + 14, :]
                        cv.imwrite('template/template_tmp/fangyuan/quan1-1.jpg', quan1_1)

    def fangtemplate(self, l_fangkuai, cvimg, cverzhi):
        '''
        方块模板选取与圆圈类似
        '''
        fangkuai = np.array(l_fangkuai).astype(np.int16)
        if len(fangkuai):
            fang_ind = np.argsort(fangkuai[:, 5])[::-1]
            fangkuai = fangkuai[fang_ind]
            num = len(fangkuai)
            step = int(num / 2)

            f3 = cverzhi[fangkuai[0][1]:fangkuai[0][3], fangkuai[0][2]:fangkuai[0][2] + 1]
            if not np.sum(f3):
                fang3 = cvimg[fangkuai[0][1]:fangkuai[0][3], fangkuai[0][0]:fangkuai[0][2], :]
                cv.imwrite('template/template_tmp/fangyuan/fang3.jpg', fang3)
            else:
                f3_1 = cverzhi[fangkuai[0][1]:fangkuai[0][3], fangkuai[0][2] + 12:fangkuai[0][2] + 13]
                if not np.sum(f3_1):
                    fang3_1 = cvimg[fangkuai[0][1]:fangkuai[0][3], fangkuai[0][0]:fangkuai[0][2] + 12, :]
                    cv.imwrite('template/template_tmp/fangyuan/fang3_1.jpg', fang3_1)

            if step != 0:
                f2 = cverzhi[fangkuai[step][1]:fangkuai[step][3], fangkuai[step][2]:fangkuai[step][2] + 1]
                if not np.sum(f2):
                    fang2 = cvimg[fangkuai[step][1]:fangkuai[step][3], fangkuai[step][0]:fangkuai[step][2], :]
                    cv.imwrite('template/template_tmp/fangyuan/fang2.jpg', fang2)
                else:
                    for m in range(50):
                        f2_1 = cverzhi[fangkuai[step][1]:fangkuai[step][3],
                               fangkuai[step][2] + m:fangkuai[step][2] + m + 1]
                        if not np.sum(f2_1):
                            fang2_1 = cvimg[fangkuai[step][1]:fangkuai[step][3],
                                      fangkuai[step][0]:fangkuai[step][2] + m + 5, :]
                            cv.imwrite('template/template_tmp/fangyuan/fang2_1.jpg', fang2_1)
                            break

            if num >= 3:
                f1 = cverzhi[fangkuai[-1][1]:fangkuai[-1][3], fangkuai[-1][2]:fangkuai[-1][2] + 1]
                if not np.sum(f1):
                    fang1 = cvimg[fangkuai[-1][1] - 2:fangkuai[-1][3] - 2, fangkuai[-1][0]:fangkuai[-1][2] + 3, :]
                    cv.imwrite('template/template_tmp/fangyuan/fang1.jpg', fang1)

                else:
                    f1_1 = cverzhi[fangkuai[-1][1]:fangkuai[-1][3], fangkuai[-1][2] + 13:fangkuai[-1][2] + 15]
                    if not np.sum(f1_1):
                        fang1_1 = cvimg[fangkuai[-1][1]:fangkuai[-1][3], fangkuai[-1][0]:fangkuai[-1][2] + 15, :]
                        cv.imwrite('template/template_tmp/fangyuan/fang1_1.jpg', fang1_1)
                    else:
                        for i in range(50):
                            f1_1_1 = cverzhi[fangkuai[-1][1]:fangkuai[-1][3],
                                     fangkuai[-1][2] + i:fangkuai[-1][2] + 1 + i]
                            if not np.sum(f1_1_1):
                                fang1_1_1 = cvimg[fangkuai[-1][1]:fangkuai[-1][3],
                                            fangkuai[-1][0]:fangkuai[-1][2] + i + 5, :]
                                cv.imwrite('template/template_tmp/fangyuan/fang1-1-1.jpg', fang1_1_1)
                                break

    def templatekuan(self, coord):
        if len(coord):
            coord_kuan = []
            for c in coord:
                w = int(c[2] - c[0])
                h = int(c[3] - c[1])
                # if (h>40) and (w<30):
                coord_kuan.append((w, h))
            coord_kuan = np.array(coord_kuan)
        else:
            coord_kuan = np.array(coord)
        return coord_kuan


if __name__ == '__main__':
    imname = 'tianzheng/yuanjuan/tz-1-12-2.jpg'
    cv_img = cv.imread(imname)
    matchtemplate = Jietemplate()
    matchtemplate.ocrtemplate(imname)
    if len(matchtemplate.left_kuo):
        for b in matchtemplate.left_kuo:
            cv.rectangle(cv_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
            cv.putText(cv_img, str(int(b[4])), (int(b[0]), int(b[1] - 10)), cv.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    if len(matchtemplate.right_kuo):
        for bk in matchtemplate.right_kuo:
            cv.rectangle(cv_img, (int(bk[0]), int(bk[1])), (int(bk[2]), int(bk[3])), (0, 255, 0), 2)
            cv.putText(cv_img, str(int(bk[4])), (int(bk[0]), int(bk[1] - 10)), cv.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0),
                       2)
    if len(matchtemplate.shiyuanquan):
        for bz in matchtemplate.shiyuanquan:
            cv.rectangle(cv_img, (int(bz[0]), int(bz[1])), (int(bz[2]), int(bz[3])), (255, 0, 0), 2)
            cv.putText(cv_img, str(int(bz[4])), (int(bz[0]), int(bz[1] - 10)), cv.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0),
                       2)
    if len(matchtemplate.l_fangkuai):
        for bf in matchtemplate.l_fangkuai:
            cv.rectangle(cv_img, (int(bf[0]), int(bf[1])), (int(bf[2]), int(bf[3])), (0, 0, 255), 2)
            cv.putText(cv_img, str(int(bf[4])), (int(bf[0]), int(bf[1] - 10)), cv.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0),
                       2)
    if len(matchtemplate.r_fangkuai):
        for bkr in matchtemplate.r_fangkuai:
            cv.rectangle(cv_img, (int(bkr[0]), int(bkr[1])), (int(bkr[2]), int(bkr[3])), (0, 0, 255), 2)
    cv.imwrite('tianzheng/yuanjuan/0.jpg', cv_img)
