'''
曹长玉
Python3.6
Pycharm2019.1
version: 2019.6.16
'''
import cv2 as cv
import numpy as np
import os
#from fill.mix.mix8.ocrjietu2 import Jietemplate
import datetime
import time

# from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageFilter
#from tesserocr import PyTessBaseAPI, PSM, iterate_level, RIL


class Template(object):
    def __init__(self):
        self.width_hang = 0
        self.width_diao = 0

    def danmatch(self, imname, match, threshold=0.65, yanse=(0, 255, 0)):
        '''
		imname为输入图片
		match为匹配模板
		图片转为灰度图
		使用模板匹配方法cv.matchTemplate()
		返回匹配对象的坐标box
		初始化1位标志位，当经过nms后标志位可能由1->0的时，该坐标将被删
        '''
        img_rgb = cv.imread(imname)
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
        template = cv.imread(match, 0)
        # print('template:',template,'\n',type(template))
        w, h = template.shape[::-1]
        res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
        threshold = threshold
        loc = np.where(res >= threshold)
        box = []
        for pt in zip(*loc[::-1]):
            box.append((pt[0], pt[1], pt[0] + w, pt[1] + h, 1))
            #cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), yanse, 1)
        # print('box:', len(box))
        return box

    def zhixian(self, imname):
        '''
        需要二值图，二值图背景为黑底背景
        使用闭运算
        查找轮廓，求出轮廓的坐标，并筛选坐标得到直线位置
        '''
        src_image = cv.imread(imname)
        src_img = cv.cvtColor(src_image, cv.COLOR_BGR2GRAY)
        ret, img = cv.threshold(src_img, 125, 255, cv.THRESH_BINARY_INV)
        element = cv.getStructuringElement(cv.MORPH_RECT, (15, 2))
        # print('cv.getStructuringElement:', element)
        #############
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, element)
        # print('cv.morphologyEx:',img,img.shape)
        #############
        h, w = img.shape[:2]

        image, contours0, hieraracy = cv.findContours(img.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = [cv.approxPolyDP(cnt, 3, True) for cnt in contours0]
        vis = np.zeros((h, w, 3), np.uint8)

        cont = []
        for cnt in contours:
            (cx1, cy1), (cw1, ch1), angle = cv.minAreaRect(cnt)
            ###
            angle1 = np.remainder(angle, 180)
            tol = 3
            is_valid = False
            if angle1 < tol or angle1 > (180 - tol):
                is_valid = True
            if angle1 < (90 + tol) and angle1 > (90 - tol):
                tmp = cw1
                cw1 = ch1
                ch1 = tmp
                is_valid = True
            ###
            # if ch1 < 100 and cw1 > 65:
            if is_valid and ch1 < 10 and cw1 > 65:
                lx = int(cx1 - cw1 / 2)
                ly = int(cy1 - ch1 / 2)
                rx = int(cx1 + cw1 / 2)
                ry = int(cy1 + ch1 / 2)
                # cv.rectangle(img,(lx+15,ry-70),(rx-15,ry-15),(255,255,255),2)
                # cv.imwrite('1.jpg',img)

                if not np.sum(img[ly - 70:ry - 7, lx + 15:rx - 15]):
                    # print('np.sum(img[ly-70:ry-10,lx+15:rx-15]):',np.sum(img[ly-70:ry-8,lx+15:rx-15]))
                    cont.append((lx, ly, rx, ry, 1))
        return cont

    def diaozhixian(self, cont):
        '''
        对查找的直线按行从大到小排列
        '''
        if len(cont) <= 1:
            return np.array(cont)

        else:
            cont = np.array(cont)
            cont_ind = np.argsort(cont[:, 3])
            cont = cont[cont_ind]

            cont_hang = []
            for ci in range(len(cont)):
                if cont[ci][4] == 0:
                    continue
                ci_hang = []
                ci_hang.append(cont[ci])
                for cj in range(ci + 1, len(cont)):
                    if abs(cont[cj][1] - cont[ci][1]) < 10:
                        cont[cj][4] = 0
                        ci_hang.append(cont[cj])

                if len(ci_hang) > 1:
                    ci_hang = np.array(ci_hang)
                    ci_ind = np.argsort(ci_hang[:, 0])
                    ci_hang = ci_hang[ci_ind]
                    cont_hang.append(ci_hang)
                else:
                    cont_hang.append(np.array(ci_hang))

            cont_hang0 = cont_hang[0]
            for conti in range(1, len(cont_hang)):
                cont_hang0 = np.vstack((cont_hang0, cont_hang[conti]))
            # cont_hang0=cont_hang0[4:,:]   ##################xiao zhu
            return cont_hang0

    def fyboxvstack(self, boxes):
        '''
        将坐标垂直拼接：
        1、判断每个模板是否匹配出坐标，若没匹配出符号坐标，danmatch返回[],将此[]变化为array[1,1,1,1,0]形式
                                       若匹配到坐标将存放每个模板匹配结果的列表转为numpy数据格式
        2、将坐标进行拼接
        3、删除标志位为0多余的坐标的坐标，返回坐标
        '''
        length = len(boxes)
        for i in range(length):
            if len(boxes[i]):
                boxes[i] = np.array(boxes[i])
            else:
                boxes[i] = np.ones((1, 5))
                boxes[i][:, 4] = 0

        box = boxes[0]
        for j in range(1, length):
            box = np.vstack((box, boxes[j]))

        boxfilter = box[:, 4] > 0.2
        boxind = np.nonzero(boxfilter)[0]
        hunbox = box[boxind].astype(np.int16)
        # print('fang yu yuan he bing de hunbox:\n', hunbox)
        return hunbox

    def nms(self, hunboxes, threshold=0.2):
        '''
        1、判断交并比是否大于0.2
        2、若大于0.2则将标志位置0
        3、删除标志位为0多余的坐标的坐标
        '''
        for n in range(len(hunboxes)):
            if hunboxes[n][4] == 0:
                continue
            for m in range(n + 1, len(hunboxes)):
                if hunboxes[m][4] == 0:
                    continue
                x1 = np.maximum(hunboxes[n][0], hunboxes[m][0])
                y1 = np.maximum(hunboxes[n][1], hunboxes[m][1])
                x2 = np.minimum(hunboxes[n][2], hunboxes[m][2])
                y2 = np.minimum(hunboxes[n][3], hunboxes[m][3])
                w = np.maximum(0, x2 - x1 + 1)
                h = np.maximum(0, y2 - y1 + 1)
                inter = w * h
                n_areas = (hunboxes[n][2] - hunboxes[n][0]) * (hunboxes[n][3] - hunboxes[n][1])
                m_areas = (hunboxes[m][2] - hunboxes[m][0]) * (hunboxes[m][3] - hunboxes[m][1])
                over = inter / (n_areas + m_areas - inter)
                # print('over:',over)
                if over >= threshold:
                    hunboxes[m][4] = 0
        hunboxes_filter = hunboxes[:, 4] > 0.2
        hunboxes_ind = np.nonzero(hunboxes_filter)[0]
        hunboxes_nms = hunboxes[hunboxes_ind]
        return hunboxes_nms

    #####################################
    def hangpipei(self, l_box, r_box):
        '''
        1、找出左右括号处在同一行的坐标：
        2、使用同一行的右括号减去左括号得到宽度，将宽度存放在列表之中
        3、对宽度列表小到大按顺序排列，使用最小宽度调整左括坐标
        4、将配对出的右括号标志位置0

        '''
        width_hang = []
        for hl_ind in range(len(l_box)):
            hang_flag = False
            ##hang_kuan yu hang_ind index wei zhi dui ying
            hang_kuan = []
            hang_ind = []
            for hr_ind in range(len(r_box)):
                #####hai xu yao pan duan r_box[4] shi fou wei 0
                # if r_box[hr_ind][4]==0:
                #    continue
                if abs((l_box[hl_ind][1] - r_box[hr_ind][1])) < 20:   #括号的高度一般大于40，右括号高度与左括好高度差值波动在20以内则认为处在同一行
                    lr_kuan = r_box[hr_ind][2] - l_box[hl_ind][0]
                    if lr_kuan > 8:  ###############
                        # print('lr_kuan',lr_kuan)
                        hang_kuan.append(lr_kuan)
                        hang_ind.append(hr_ind)
                        width_hang.append(lr_kuan)
                        hang_flag = True
            # print('width_diao:',hang_kuan)
            if hang_flag:
                # print('lr_kuan', lr_kuan)
                hang_kuan = np.array(hang_kuan)
                kuan_index = np.argsort(hang_kuan)
                ind_kuan = hang_kuan[kuan_index[0]]
                l_box[hl_ind][0] = l_box[hl_ind][0] + 25
                l_box[hl_ind][2] = l_box[hl_ind][0] + ind_kuan - 50
                hang_ind = np.array(hang_ind)
                r_box[hang_ind[kuan_index[0]]][4] = 0

        return l_box, r_box

    #######################################################
    def diaokuokuan(self, l_box, r_box, threshold=0.1):
        '''
        判断模板匹配出的是否匹配出左括号与右括号
        若模板匹配同时未成功匹配出左右括号，返回一个空列表
        若模板匹配匹配出左右括号：
            1、当左右括号数量相同时，使用hangpipei函数匹配左右括号对，
                根据真实宽度调整左括号，筛除右括号
            2、当左右括号数量不同时，使用交并比判断是否成对出现。
                成对出现则用真实宽度调整总括号，若不成对出现在使用130去调整
        删除标志位为0多余的坐标
        '''
        # print('left shu liang:', len(l_box))
        # print('right shu liang:', len(r_box))
        if (l_box[0, 0] == 1) and (r_box[0, 0] == 1):
            kuohao_box = []
            return np.array(kuohao_box)

        if len(l_box) == len(r_box):
            if (len(l_box) == 1) and (len(r_box) == 1):
                box_kuan = int(r_box[0, 2] - l_box[0, 0])
                l_box[0, 0] = l_box[0, 0] + 25
                l_box[0, 2] = l_box[0, 0] + box_kuan - 50
                r_box[:, 4] = 0
            else:
                l_box, r_box = self.hangpipei(l_box, r_box)
            ##########################################################
            kuohao = np.vstack((l_box, r_box))
            filter = kuohao[:, 4] > 0.2
            ind = np.nonzero(filter)[0]
            kuohao_box = kuohao[ind]
            ##########################################################

        else:
            width_diao = []
            l_box[:, 2] = l_box[:, 2] + 60
            r_box[:, 0] = r_box[:, 0] - 80
            for l in l_box:
                flag = False
                for r in r_box:
                    xx1 = np.maximum(l[0], r[0])
                    yy1 = np.maximum(l[1], r[1])
                    xx2 = np.minimum(l[2], r[2])
                    yy2 = np.minimum(l[3], r[3])
                    w1 = np.maximum(0, xx2 - xx1 + 1)
                    h1 = np.maximum(0, yy2 - yy1 + 1)
                    inter1 = w1 * h1
                    # print('inter1:', inter1)
                    l_areas1 = (l[2] - l[0]) * (l[3] - l[1])
                    r_areas1 = (r[2] - r[0]) * (r[3] - r[1])
                    over1 = inter1 / (l_areas1 + r_areas1 - inter1)
                    # print('over1:',over1)
                    if over1 >= threshold:
                        r[4] = 0
                        flag = True
                        kuan = int(r[2] - l[0])
                        width_diao.append(kuan)
                if flag:
                    l[0] = l[0] + 25
                    l[2] = l[0] + width_diao[-1] - 50
                else:
                    l[0] = l[0] + 10
                    l[2] = l[0] + 130
            #############################################################
            kuohao = np.vstack((l_box, r_box))
            filter = kuohao[:, 4] > 0.2
            ind = np.nonzero(filter)[0]
            kuohao_box = kuohao[ind]
            kuohao_box = self.nms(kuohao_box, threshold=0.1)
            ###########################################################
        '''
        kuohao=np.vstack((l_box,r_box))
        filter=kuohao[:,4]>0.2
        ind=np.nonzero(filter)[0]
        kuohao_box=kuohao[ind]
        '''
        return kuohao_box

    def totalmatch(self, imname):
        '''
        1、获取方块与圆圈模板文件夹路径，使用多模板匹配
        2、获取左括号模板文件夹路径，使用多模板匹配
        3、获取右括号模板文件夹路径，使用多模板匹配

        4、方块圆圈调用fyboxvstack函数，实现多模板匹配坐标的拼接
        5，方块与圆圈通过调用nms（）函数过滤重复的框，并对结果进行判断。若nms()后返回为空，则将此返回结果转为array([1,1,1,1,0])形式

        6、左括号调用fyboxvsatck（）函数，实现多模板匹配坐标的拼接
        7、调用nms函数过滤重复的框，并对结果进行判断。若nms()后返回为空，则将此返回结果转为array([1,1,1,1,0])形式

        8、左括号调用fyboxvsatck（）函数，实现多模板匹配坐标的拼接
        9、调用nms函数过滤重复的框，并对结果进行判断。若nms()后返回为空，则将此返回结果转为array([1,1,1,1,0])形式

        10、调用diaokuokuan（）函数实现括号宽度的调整，及删除右括号坐标
        11、调用zhixian（）函数及diaozhixian()函数实现直线位置定位
        12，将方块圆圈、括号、直线位置进行拼接。方块圆圈标志位为0，括号标志位为1，直线标志位为2

        '''
        match_path_fangyuan = os.path.join('template', 'template_tmp', 'fangyuan')  ##'template/template_tmp/fangyuan'
        match_image = os.listdir(match_path_fangyuan)
        boxes_fangyuan = []
        for match_img in match_image:
            match = os.path.join(match_path_fangyuan, match_img)  ##'template/template_tmp/fang.jpg'
            box = self.danmatch(imname, match)
            # box=self.danmatch(imname,match,threshold=0.8)
            boxes_fangyuan.append(box)
        # print('boxes_fangyuan:', boxes_fangyuan)

        match_kuohao = os.path.join('template', 'template_tmp', 'kuohao')  ##'template/template_tmp/kuohao'
        match_leftbox_image = os.listdir(match_kuohao + '/leftkuo')
        boxes_left = []
        kuotime1 = time.time()
        for match_leftbox_img in match_leftbox_image:
            left_match = os.path.join(match_kuohao, 'leftkuo', match_leftbox_img)
            box_left = self.danmatch(imname, left_match, threshold=0.8)
            boxes_left.append(box_left)
        kuotime2 = time.time()
        # print('kuotime:', int(kuotime2 - kuotime1))
        # print('boxes_left:', boxes_left)

        match_rightbox_image = os.listdir(match_kuohao + '/rightkuo')
        boxes_right = []
        for match_rightbox_img in match_rightbox_image:
            right_match = os.path.join(match_kuohao, 'rightkuo', match_rightbox_img)
            box_right = self.danmatch(imname, right_match, threshold=0.8)
            boxes_right.append(box_right)
        # print('boxes_right:', boxes_right)

        hunbox = self.fyboxvstack(boxes_fangyuan)
        if len(hunbox):
            hunbox_fangyuan = self.nms(hunbox)
            if not len(hunbox_fangyuan):#######当nms返回为空时，将返回结果改为array([1,1,1,1,0])形式
                hunbox_fangyuan = np.ones((1, 5))
                hunbox_fangyuan[:, 4] = 0
        else:  ###########当fyboxvstack（）函数返回为空时，将返回结果改为array([1,1,1,1,0])形式
            hunbox_fangyuan = np.ones((1, 5))
            hunbox_fangyuan[:, 4] = 0

        hunbox_left = self.fyboxvstack(boxes_left)
        if len(hunbox_left):
            hunbox_left = self.nms(hunbox_left)
            if not len(hunbox_left):
                hunbox_left = np.ones((1, 5))
                hunbox_left[:, 4] = 0
        else:
            hunbox_left = np.ones((1, 5))
            hunbox_left[:, 4] = 0

        hunbox_right = self.fyboxvstack(boxes_right)
        if len(hunbox_right):
            hunbox_right = self.nms(hunbox_right)
            if not len(hunbox_right):
                hunbox_right = np.ones((1, 5))
                hunbox_right[:, 4] = 0
        else:
            hunbox_right = np.ones((1, 5))
            hunbox_right[:, 4] = 0

        kuohao = self.diaokuokuan(hunbox_left, hunbox_right)
        hunbox_fangyuan = hunbox_fangyuan[np.nonzero(hunbox_fangyuan[:, 4])[0]]

            ###第6位用来区分方块圆圈（0），括号（1），直线（2）
        kuohao_flag = np.ones((kuohao.shape[0], 6))
        kuohao_flag[:, :5] = kuohao
        hunbox_fangyuan_flag = np.zeros((hunbox_fangyuan.shape[0], 6))
        hunbox_fangyuan_flag[:, :5] = hunbox_fangyuan

        zhixian_box = self.zhixian(imname)
        zhixian_box_hang = self.diaozhixian(zhixian_box)###对直线排序

        if len(zhixian_box_hang):
            zhixian_box_hang_flag = np.ones((zhixian_box_hang.shape[0], 6)) + 1
            zhixian_box_hang_flag[:, :5] = zhixian_box_hang
            total_box = np.vstack((hunbox_fangyuan_flag, kuohao_flag, zhixian_box_hang_flag))
        else:
            total_box = np.vstack((hunbox_fangyuan_flag, kuohao_flag))
        return total_box

        # return hunbox_fangyuan,kuohao,zhixian_box_hang


if __name__ == '__main__':
    '''
    imname='tianzheng/yuanjuan/tz-1-12-1.jpg'
    cv_img=cv.imread(imname)
    a=time.time()
    ocrjie=Jietemplate()
    ocrjie.ocrtemplate(imname)
    print('ocrtemplate time:',int(time.time()-a))
    matchtemplate=Template()
    box_fangyuan,box_kuohao,box_zhixian=matchtemplate.totalmatch(imname)
    b=time.time()
    print('time:',int(b-a))
    print('box_fangyuan:',box_fangyuan,'\n',len(box_fangyuan))
    print('box_kuohao:',box_kuohao,'\n',len(box_kuohao))
    print('box_zhixian:',box_zhixian,'\n',len(box_zhixian))

    if len(box_fangyuan):
        for b in box_fangyuan:
            cv.rectangle(cv_img,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),(0,255,0),2)

    if len(box_kuohao):
        for bk in box_kuohao:
            cv.rectangle(cv_img,(int(bk[0]),int(bk[1])),(int(bk[2]),int(bk[3])),(0,255,0),2)

    if len(box_zhixian):
        for bz in box_zhixian:
            cv.rectangle(cv_img,(int(bz[0]),int(bz[1])),(int(bz[2]),int(bz[3])),(255,0,0),2)
    
    cv.imwrite('shiyan/2019613/0.jpg',cv_img)


    #cv.imwrite('tianzheng/shiyan/2-2.jpg', cv_img)
    #cv.imshow('cv_img',cv_img)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    
    left_template='template/yuanquan.jpg'
    for name in imname:
        name_path=os.path.join('yuanjuan',name)
        print('name_path:',name_path)
        matchtemplate.match(name_path,left_template)
    '''

    imname = 'tianzheng/yuanjuan/tz-1-12-2.jpg'
    cv_img = cv.imread(imname)
    matchtemplate = Template()
    total = matchtemplate.totalmatch(imname)
    if len(total):
        for b in total:
            cv.rectangle(cv_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
    cv.imwrite('template/2.jpg', cv_img)

    '''
    imnames=os.listdir('tianzheng/yuanjuan')
    for name in imnames:
        imname=os.path.join('tianzheng','yuanjuan',name)
        cv_img = cv.imread(imname)
        a = time.time()
        ocrjie = Jietemplate()
        ocrjie.ocrtemplate(imname)
        #print('ocrtemplate time:', int(time.time() - a))
        matchtemplate = Template()
        box_fangyuan, box_kuohao, box_zhixian = matchtemplate.totalmatch(imname)
        b = time.time()
        #print('time:', int(b - a))
        #print('box_fangyuan:', box_fangyuan, '\n', len(box_fangyuan))
        #print('box_kuohao:', box_kuohao, '\n', len(box_kuohao))
        #print('box_zhixian:', box_zhixian, '\n', len(box_zhixian))

        if len(box_fangyuan):
            for b in box_fangyuan:
                cv.rectangle(cv_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)

        if len(box_kuohao):
            for bk in box_kuohao:
                cv.rectangle(cv_img, (int(bk[0]), int(bk[1])), (int(bk[2]), int(bk[3])), (0, 255, 0), 2)

        if len(box_zhixian):
            for bz in box_zhixian:
                cv.rectangle(cv_img, (int(bz[0]), int(bz[1])), (int(bz[2]), int(bz[3])), (255, 0, 0), 2)

        cv.imwrite('shiyan/2019613/'+name, cv_img)
    '''
