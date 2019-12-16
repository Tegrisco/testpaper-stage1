"""
=====================================
Get the coordinates of DaTi and XiaoTi
=====================================

A script for Python3.

Use Tesseract OCR and template matching to locate the kousuanti region.
结合Tesseract OCR 和模版匹配定位大题和小题区域。

"""


# Add Python3.6 site-packages, timing tasks required
import sys
sys.path.append('/usr/local/Python3.6/lib/python3.6/site-packages')
# import AUTO_OSD

# Python Imaging Library
from PIL import Image

# Numeric Python
import numpy as np

# opencv
import cv2

# Regular expression 
import re

# Stack traces, debug required
import traceback

# Deep copy
import copy

# Switch locale require
import locale
from contextlib import contextmanager


class Paper():
    """Main class which contains top level functions.
    
    Fetches titles and regions for Dati and XiaoTi.
    Matching question lists.

    Attributes:
        imagePath: A string of image file path.
        questionListDict: A dic contains question list.
        imagePIL: A PIL image object.
        imageCV2: An opencv image object.
        width: An integer of the width of paper image.
        height: An integet of the height of paper image.
        coorOfDaTiTitles: A dic of DaTi titles' coordinates.
        coorOfDaTiPartitions: A dic of DaTi partitions' coordinates.
        coorOfXiaoTiTitles: A dic of XiaoTi titles' coordinates.
        coorOfXiaoTiPartitions: A dic of XiaoTi partitions' coordinates.

    Functions:
        c_locale(): Switches to C locale.
        get_dati_titles(): Gets DaTi titles' coordinates.
        get_dati_partitions(): Gets DaTi partitions' coordinates.
        get_xiaoti_titles(): Gets XiaoTi titles' coordinates.
        get_xiaoti_partitions(questionListDict): Gets XiaoTi partitions' 
        coordinates.
        match_questionList(): Matches questions list.

    PS:
        Each function will be called in order:
        dati_title -> dati_partition -> xiaoti_title -> xiaoti_partition
        Because they are depend on the results before. For example:
        If you want to get the coorOfDaTiPartitions and call 
        get_dati_partitions() but coorOfDaTiTitles is empty, it will 
        automatically call get_dati_titles().

    """
    imagePath = ''
    questionListDict = {}
    imagePIL = []
    imageCV2 = np.array([])
    width = 0
    height = 0
    coorOfDaTiTitles = {}
    coorOfDaTiPartitions = {}
    coorOfXiaoTiTitles = {}
    coorOfXiaoTiPartitions = {}

    
    def __init__(self, i):
        self.imagePath = i
        #self.questionListDict = q
        self.imageCV2 = cv2.imread(i)
        self.imagePIL = Image.open(i)
        self.width = self.imageCV2.shape[1]
        self.height = self.imageCV2.shape[0]


    @contextmanager
    def c_locale(self):
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


    #Get titles position
    def get_dati_titles(self):
     
        coorOfDaTiTitles = {}
        pattern = re.compile(u"一 、|一、")

        wholePage = WholePage(self.imagePath,[])
        #print ('Ready to enter c_locale')
        with self.c_locale():
            from tesserocr import PSM
            wholePageText = wholePage.getText(PSM.AUTO_OSD, 'chi_sim', True, False, False)
        #get total DaTi in one page
        minNum = 1 if pattern.search(wholePageText) else match_dati_positive(wholePageText)
        maxNum = match_dati_reverse(wholePageText) if pattern.search(wholePageText) else match_dati_reverse(wholePageText) + 1
        totalNum = maxNum - minNum + 1

        #Divide to two sides
        leftPage = WholePage(self.imagePath, [0, 0, int(self.width / 2), self.height])
        rightPage = WholePage(self.imagePath, [int(self.width / 2), 0, self.width, self.height])
        with self.c_locale():
            from tesserocr import RIL, PSM
            coorOfDaTiTitles.update(leftPage.getCoordinate(PSM.AUTO_OSD, 'chi_sim', RIL.TEXTLINE, 0, True, True, False))
            coorOfDaTiTitles.update(rightPage.getCoordinate(PSM.AUTO_OSD, 'chi_sim', RIL.TEXTLINE, 0, True, True, False))

        if (len(coorOfDaTiTitles) < totalNum):
            with self.c_locale():
                from tesserocr import RIL, PSM
                tempCoor = leftPage.getCoordinate(PSM.AUTO_OSD, 'chi_sim', RIL.TEXTLINE, 0, False, True, False)
                tempCoor.update(rightPage.getCoordinate(PSM.AUTO_OSD, 'chi_sim', RIL.TEXTLINE, 0, False, True, False))
            for i,j in tempCoor.items():
                if i not in coorOfDaTiTitles:
                        coorOfDaTiTitles[i] = j

        if (len(coorOfDaTiTitles) < totalNum):
            with self.c_locale():
                from tesserocr import RIL, PSM
                tempCoor = wholePage.getCoordinate(PSM.AUTO_OSD, 'chi_sim', RIL.TEXTLINE, 0, True, False, False)
            for i,j in tempCoor.items():
                if i not in coorOfDaTiTitles:
                        coorOfDaTiTitles[i] = j

        if (len(coorOfDaTiTitles) < totalNum):
            with self.c_locale():
                from tesserocr import RIL, PSM
                tempCoor = wholePage.getCoordinate(PSM.SPARSE_TEXT_OSD, 'chi_sim', RIL.TEXTLINE, 0, True, False, False)
            for i,j in tempCoor.items():
                if i not in coorOfDaTiTitles:
                        coorOfDaTiTitles[i] = j

        keys = list(coorOfDaTiTitles.keys())
        if 0 in coorOfDaTiTitles :
            # FuJiaTi should not count as zero
            coorOfDaTiTitles[max(keys) + 1] = coorOfDaTiTitles[0]
            del(coorOfDaTiTitles[0])

        #print ('coorOfDaTiTitles is:\n{}'.format(coorOfDaTiTitles))
        self.coorOfDaTiTitles = coorOfDaTiTitles
        #return coorOfDaTiTitles


    #Get dati partitions
    def get_dati_partitions(self):

        if len(self.coorOfDaTiTitles) == 0 :
            self.get_dati_titles()

        coorOfDaTiTitles = copy.deepcopy(self.coorOfDaTiTitles)
        halfWidth = int(self.width / 2)
        keys = list(coorOfDaTiTitles.keys())
        
        coorOfDaTiPartitions = {}
        titles = []
        page = 0 if 1 in coorOfDaTiTitles else 1
        
        maxNum = max(keys)
        minNum = min(keys)

        # Make titles class
        for i,j in coorOfDaTiTitles.items():
            mark = str(i)
            side = 1 if j[0] < halfWidth else 2
            nextKey = None if i + 1 not in keys else i + 1
            isMin = False if i != minNum else True
            isMax = False if i != maxNum else True
            isMid = False if (isMax and j[0] >= halfWidth) or (nextKey and \
                (j[0] - halfWidth) * (coorOfDaTiTitles[nextKey][0] - halfWidth) >= 0) else True
            titles.append(DaTiTitle(mark, side, nextKey, isMin, isMax, isMid, j))

        try:

            for t in titles :
                if page == 1 and t.isMin and t.nextKey != None:
                    coorOfDaTiPartitions[str(int(t.mark) - 1) + '-2'] = [0, 0, halfWidth, t.position[1]]
                    if t.isMid :
                        coorOfDaTiPartitions[t.mark + '-1'] = [0, t.position[3], halfWidth, self.height]
                        coorOfDaTiPartitions[t.mark + '-2'] = [halfWidth, 0, self.width, coorOfDaTiTitles[t.nextKey][1]]
                    else :
                        coorOfDaTiPartitions[t.mark] = [0, t.position[3], halfWidth, coorOfDaTiTitles[t.nextKey][1]]
                elif t.isMid :
                    coorOfDaTiPartitions[t.mark + '-1'] = [0, t.position[3], halfWidth, self.height]
                    if t.isMax :
                        coorOfDaTiPartitions[t.mark + '-2'] = [halfWidth, 0, self.width, self.height]
                    else :
                        coorOfDaTiPartitions[t.mark + '-2'] = [halfWidth, 0, self.width, coorOfDaTiTitles[t.nextKey][1]]
                elif t.isMax :
                    if page == 0 :
                        coorOfDaTiPartitions[t.mark + '-1'] = [halfWidth, t.position[3], self.width, self.height]
                    else :
                        coorOfDaTiPartitions[t.mark] = [halfWidth, t.position[3], self.width, self.height]
                elif t.side == 1 :
                    coorOfDaTiPartitions[t.mark] = [0, t.position[3], halfWidth, coorOfDaTiTitles[t.nextKey][1]]
                elif t.side == 2 :
                        coorOfDaTiPartitions[t.mark] = [halfWidth, t.position[3], self.width, coorOfDaTiTitles[t.nextKey][1]]

        except KeyError:
            traceback.print_exc()
            print ('Oops! There is an KeyError while dividing DaTi {}, skip this part...'.format(t.mark))

        # 判断区域是否空白
        image = self.imageCV2.copy()
        def region_is_empty(coor):
            c = coor[1]
            return not(image[c[1]:c[3], c[0]:c[2]].all())

        #print ('coorOfDaTiPartitions is:\n{}'.format(coorOfDaTiPartitions))
        self.coorOfDaTiPartitions = dict(filter(region_is_empty, coorOfDaTiPartitions.items()))
        #return coorOfDaTiPartitions


    def get_xiaoti_titles(self):

        if len(self.coorOfDaTiPartitions) == 0 :
            self.get_dati_partitions()
        
        coorOfDaTiPartitions = copy.deepcopy(self.coorOfDaTiPartitions)
        coorOfXiaoTiTitles = {}    #整张试卷最后的题号坐标，例：{'1':{1:[1,2,3,4]}}
        datiPartitions = []    #大题区域类列表

        indentDatiLeft = self.width  #左侧大题缩进
        indentDatiRight = self.width #右侧大题缩进
        for i,j in self.coorOfDaTiTitles.items() :
            if j[0] < int(self.width / 2) and j[0] < indentDatiLeft:
                indentDatiLeft = j[0] - 10
            elif j[0] >= int(self.width / 2) and j[0] < indentDatiRight:
                indentDatiRight = j[0] - int(self.width / 2) - 10

        if indentDatiLeft == self.width : indentDatiLeft = 0
        if indentDatiRight == self.width : indentDatiRight = 0
        #大题区域右边界为大题题号
        for i,j in coorOfDaTiPartitions.items():
            
            if j[0] == 0 :
                position = [indentDatiLeft] + j[1:4]
            else :
                position = [indentDatiRight + j[0]] + j[1:4]
            datiPartitions.append(DaTiPartition(self.imagePath, position, i))

        for d in datiPartitions :
            #print ('datiPartition: {}'.format(d.mark))
            coorOfXiaoTiTitlesInPartition = {}
            with self.c_locale():
                from tesserocr import PSM
                text = d.getText(PSM.SINGLE_BLOCK, 'chi_sim', True, True, False)
            if d.isEmpty :
                if d.position[1] != 0 :
                    coorOfXiaoTiTitles[d.mark] = {0: d.position}         
                continue

            coorOfXiaoTiTitlesInPartition[0] = d.position
            with self.c_locale():
                from tesserocr import PSM, RIL
                coorOfXiaoTiTitlesInPartition.update(d.getCoordinate(PSM.SINGLE_BLOCK, 'chi_sim', 
                                                                    RIL.TEXTLINE, 20, True, True, False))

                tempCoor = d.getCoordinate(PSM.SINGLE_BLOCK, 'chi_sim', RIL.TEXTLINE, 20, False, False, True)
            for i,j in tempCoor.items():
                if i not in coorOfXiaoTiTitlesInPartition :
                    coorOfXiaoTiTitlesInPartition[i] = \
                    [j[0] + d.position[0], j[1] + d.position[1], j[2] + d.position[0], j[3] + d.position[1]]
            
            coorOfXiaoTiTitles[d.mark] = coorOfXiaoTiTitlesInPartition

        #print ('ocr-coorOfXiaoTiTitles is:\n{}'.format(coorOfXiaoTiTitles))
        self.coorOfXiaoTiTitles = coorOfXiaoTiTitles

        if len(self.questionListDict) != 0 :
            
            self.match_questionList()

        #return coorOfXiaoTiTitles


    def match_questionList(self):

        questionList = analyzeAnswerList(self.questionListDict)
        coorOfXiaoTiTitles = copy.deepcopy(self.coorOfXiaoTiTitles)

        tempCoor = {}
        missXiaoTi = {}
        for i,j in coorOfXiaoTiTitles.items():
            tempCoor[int(i[0])] = {}
            missXiaoTi[int(i[0])] = []

        copyOfCoor = copy.deepcopy(coorOfXiaoTiTitles)
        for i,j in copyOfCoor.items():
            if len(j) == 1 and questionList[int(i[0])] == 1:
                j[1] = j[0]
            del j[0]
            tempCoor[int(i[0])].update(j) 
        #print ('questionListR = \n{}'.format(tempCoor))

        for i,j in tempCoor.items():
            if len(j) != questionList[i]:
                for n in range(1, questionList[i] + 1):
                    if n not in j:
                        missXiaoTi[i].append(n)
            else:
                del missXiaoTi[i]
        
        #print ('point1 coorOfXiaoTiTitles is:\n{}'.format(coorOfXiaoTiTitles))

        if len(missXiaoTi) != 0:
            #print ('ocr missXiaoTi = \n{}'.format(missXiaoTi))
            modelImage = {}
            missValues = []
            
            for i,j in missXiaoTi.items():
                missValues += j

            for i in missValues:
                for k,l in coorOfXiaoTiTitles.items():
                    if i in l:
                        
                        modelImage[i] = self.imagePIL.crop((l[i][0], l[i][1], l[i][0] + 80, l[i][3]))
                        #modelImage[i].show()
            
            for i,j in missXiaoTi.items():
                for k,l in coorOfXiaoTiTitles.items():
                    if int(k[0]) == i :
                        datiPil = self.imagePIL.crop(tuple(l[0]))
                        #datiPil.show()
                        
                        for m in j.copy():
                            coor = []
                            if m in modelImage:
                                template = modelImage[m]
                                coor = template_match(datiPil, template, 0.8)
                                if len(coor) != 0:
                                    coorOfXiaoTiTitles[k].update({m:[coor[0] + l[0][0], coor[1] + l[0][1], \
                                    coor[2] + l[0][0], coor[3] + l[0][1]]})
                                    j.remove(m)
            
            for i in missXiaoTi.copy():
                if len(missXiaoTi[i]) == 0:
                    del missXiaoTi[i]

        #print ('point2 coorOfXiaoTiTitles is:\n{}'.format(coorOfXiaoTiTitles))
        if len(missXiaoTi) != 0:
            dotImage = 0
            boxs = {}
            conf = 0
            for i,j in coorOfXiaoTiTitles.items():
                for k,l in j.items():
                    if k == 0 or k == 2 or k == 4: continue
                    #if dotImage != 0 and dotImage.height < l[3] - l[1]: continue
                    
                    r = Region(self.imagePath, l)
                    tempImage = self.imagePIL.crop(tuple(l))
                    #tempImage.show()
                    with self.c_locale():
                        from tesserocr import RIL, PSM
                        text = r.getText(PSM.SINGLE_LINE, 'chi_sim', True, True, False)
                    if r.isEmpty: continue
                    with self.c_locale():
                        from tesserocr import RIL, PSM
                        boxs = r.getCoordinate(PSM.SINGLE_LINE, 'chi_sim', RIL.SYMBOL, 5, True, True, False)
                    for n,m in boxs.items():
                        if m['box'][2] - m['box'][0] < 10 : continue
                        if re.match(u'(\.|、)', n):
                            if conf != 0 and conf > m['conf'] : break
                            upperBound = m['box'][3] - 70 if m['box'][3] - l[1] > 70 else l[1]
                            #leftBound = m['box'][0] + 3 if k == 4 or k == 2 else m['box'][0]
                            
                            dotImage = self.imagePIL.crop((m['box'][0], upperBound, m['box'][2], m['box'][3]))
                            conf = m['conf']
                            #dotImage.show()
                            #tempImage.show()
                            break
            #tempImage.show()
            #dotImage.show()

            for i,j in missXiaoTi.items():
                for k,l in coorOfXiaoTiTitles.items():
                    if int(k[0]) == i :
                        #print ('In {}'.format(k))
                        
                        #keys = list(l.keys())
                        #keys.remove(0)
                        #minNum = min(keys)
                        #maxNum = max(keys)
                        for m in j.copy():
                            #print ('Finding {} ...'.format(m))
                            subRegion = l[0].copy()
                            if m == 1 and 2 not in j : continue
                            
                            previous = m - 1
                            while previous not in l :
                                previous -= 1
                            
                            next = m + 1
                            while next in j :
                                next += 1
                            #print ('previous = {}'.format(previous))
                            #print ('next = {}'.format(next))
                            #if previous not in l and next not in l : continue
                            if (m - 1 not in j and m - 1 not in l and previous != 0): 
                                #print ('It\'s sure that {} is not in this region.'.format(m))
                                continue

                            coor = []
                            if previous in l and previous != 0 :
                                subRegion[1] = l[previous][3]
                            if next in l :
                                subRegion[3] = l[next][1]
                            if subRegion[1] >= subRegion[3] : 
                                #print ('Invalid region, skip...')
                                continue
                            subImage = self.imagePIL.crop(tuple(subRegion))
                            #subImage.show()
                            r = Region(self.imagePath, subRegion)
                            with self.c_locale():
                                from tesserocr import RIL, PSM
                                text = r.getText(PSM.SINGLE_BLOCK, 'chi_sim', True, True, False)
                            if r.isEmpty: 
                                #print ('subRegion is empty!')
                                continue
                            
                            if dotImage.height > subImage.height :
                                #print ('The height of subRegion is less then template, skip...')
                                continue
                            threshold = 0.81 if previous == m - 1 and next == m + 1 and next in l else 0.85

                            coor = template_match(subImage, dotImage, threshold)
                            if len(coor) != 0:
                                if m == 1 and 2 in j :
                                    subImage = self.imagePIL.crop((subRegion[0], subRegion[1], \
                                    subRegion[2], coor[1] + subRegion[1]))
                                    coor1 = template_match(subImage, dotImage, threshold)
                                    if len(coor1) != 0 :
                                        coor = coor1
                                #print ('Finded {}'.format(m))
                                coorOfXiaoTiTitles[k].update({m:[coor[0] + subRegion[0], 
                                                            coor[1] + subRegion[1], coor[2] + subRegion[0], 
                                                            coor[3] + subRegion[1]]})
                                j.remove(m)
                                
        self.coorOfXiaoTiTitles = coorOfXiaoTiTitles
        #print ('coorOfXiaoTiTitles is:\n{}'.format(coorOfXiaoTiTitles))
        
        #return coorOfXiaoTiTitles

    def get_xiaoti_partitions(self, questionListDict):

        if len(self.coorOfXiaoTiTitles) == 0 :
            self.questionListDict = questionListDict
            self.get_xiaoti_titles()

        coorOfXiaoTiPartitions = {}
        coorOfXiaoTiTitles = copy.deepcopy(self.coorOfXiaoTiTitles)

        for i,j in coorOfXiaoTiTitles.items():
            for k,l in j.items():
                coorOfXiaoTiTitles[i][k][0] = j[0][0]
                coorOfXiaoTiTitles[i][k][2] = j[0][2]

        for i,j in coorOfXiaoTiTitles.items():
            titles = {}
            coorOfXiaoTiPartitionsInDati = {}
            #找到排除掉[0]（即大题区域）之外的最小小题和最大小题
            keys = list(j.keys())
            keys.remove(0)
            #如果除[0]（即大题区域）之外没有找到小题
            if len(keys) == 0:
                coorOfXiaoTiPartitions[i[0]] = {1: j[0]}
                continue

            maxNum = max(keys)
            minNum = min(keys)
            
            #为每个小题生成对象
            keys.sort()
            try:
                for k in keys:
                    mark = k
                    nextKey = None if k + 1 not in keys else k + 1
                    isMin = False if k != minNum else True
                    isMax = False if k != maxNum else True
                    position = j[k]
                    #print('point1')
                    titles[mark] = XiaoTiTitle(mark, nextKey, isMin, isMax, position)

                for m,t in titles.items():
                    if t.mark != 1 and (len(titles) == 1 or t.isMin):
                        #print('point2')
                        p = [j[0][0], j[0][1], j[0][2], t.position[1]]
                        r = Region(self.imagePath, p)
                        with self.c_locale():
                            from tesserocr import PSM
                            text = r.getText(PSM.SINGLE_BLOCK, 'chi_sim', True, True, False)
                        if not r.isEmpty:
                            coorOfXiaoTiPartitionsInDati[t.mark - 1] = p
                    if t.isMax:
                        coorOfXiaoTiPartitionsInDati[t.mark] = t.position[0:3] + [j[0][3]]
                    elif t.nextKey != None:
                        coorOfXiaoTiPartitionsInDati[t.mark] = t.position[0:3] + \
                        [titles[t.nextKey].position[1]]
            
            except KeyError:
                traceback.print_exc()
                print ('dati = {}'.format(i))
                print ('keys = {}'.format(keys))
                print ('coorOfXiaoTiPartitionsInDati = \n{}'.format(coorOfXiaoTiPartitionsInDati))
            
            coorOfXiaoTiPartitions[i] = coorOfXiaoTiPartitionsInDati
        #print ('coorOfXiaoTiPartitions = \n{}'.format(coorOfXiaoTiPartitions))

        #将字典键值转化为整型并进行整合排序
        keys = list(coorOfXiaoTiPartitions.keys())
        keys.sort(reverse = True)
        tempCoor = {}
        for i in keys:
            tempCoor[int(i[0])] = {}
        for i,j in coorOfXiaoTiPartitions.items():
            for k,l in j.items():
                if k in tempCoor[int(i[0])] :
                    tempCoor[int(i[0])][k] += l
                else :
                    tempCoor[int(i[0])][k] = l
        #将附加题编号改为最大值
        if '0' in coorOfXiaoTiPartitions:
            tempCoor[int(max(keys)[0]) + 1] = tempCoor[0]
            del tempCoor[0]

        coorOfXiaoTiPartitions = tempCoor

        #print ('coorOfXiaoTiPartitions is:\n{}'.format(coorOfXiaoTiPartitions))
        self.coorOfXiaoTiPartitions = coorOfXiaoTiPartitions
        #return coorOfXiaoTiPartitions


class Region():
    """Tesseract OCR recognizes the region.

    Attributes:
        imagePath: A string of image file path.
        position: An array of region's coordinate.
        isEmpty: A boolean indicating if this region is empty.
        width: An integer of the width of paper image.
        height: An integet of the height of paper image.
        imagePIL: A PIL image object.
        imageCV2: An opencv image object.

    Functions:
        c_locale(): Switches to C locale.
        getText(): Gets text in this region.
        getCoordinate(): Gets texts and their boxes and confidence.
        coorFilter(): Filter the coordinate by regular expression.

    """
    imagePath = ''
    position = []
    isEmpty = False
    width = 0
    height = 0
    imageCV2 = np.array([])
    imagePIL = []

    def __init__(self, i, p):
        self.imagePath = i
        self.position = p
        self.imageCV2 = cv2.imread(i)
        self.imagePIL = Image.open(i)
        self.width = self.imageCV2.shape[1]
        self.height = self.imageCV2.shape[0]

    @contextmanager
    def c_locale(self):
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


    def getText(self, pageSegMode, languagePackage, setThreshold, setRectangle,
                setCropImage):
        with self.c_locale():
            from tesserocr import PyTessBaseAPI
            with PyTessBaseAPI(psm = pageSegMode, lang = languagePackage) as api:
                api.SetImageFile(self.imagePath)
                if setThreshold :
                    thresholdedImage=api.GetThresholdedImage()
                    api.SetImage(thresholdedImage)
                if setRectangle :
                    w = self.position[2] - self.position[0]
                    h = self.position[3] - self.position[1]
                    api.SetRectangle(self.position[0], self.position[1], w, h)
                elif setCropImage :
                    cropImage = self.imagePIL.crop(tuple(self.position))
                    api.SetImage(cropImage)
                api.Recognize()
                text = api.GetUTF8Text()
                if text == '' :
                    self.isEmpty = True
        
        return text

    def getCoordinate(self, pageSegMode, languagePackage, level, boxPadding, 
                        setThreshold, setRectangle, setCropImage):
        coordinate = {}

        with self.c_locale():
            from tesserocr import PyTessBaseAPI

            with PyTessBaseAPI(psm = pageSegMode, lang = languagePackage) as api:
                api.SetImageFile(self.imagePath)
                if setThreshold :
                    thresholdedImage=api.GetThresholdedImage()
                    api.SetImage(thresholdedImage)
                if setRectangle :
                    w = self.position[2] - self.position[0]
                    h = self.position[3] - self.position[1]
                    api.SetRectangle(self.position[0], self.position[1], w, h)
                elif setCropImage :
                    cropImage = self.imagePIL.crop(tuple(self.position))
                    api.SetImage(cropImage)
                api.Recognize()         
                ri = api.GetIterator()
                with self.c_locale():
                    from tesserocr import iterate_level
                    for r in iterate_level(ri, level):
                        try:
                            text = r.GetUTF8Text(level)
                        except RuntimeError as e:
                            # api.End()
                            print(e)
                            return coordinate
                        conf = r.Confidence(level)
                        box = r.BoundingBox(level, boxPadding)
                        coordinate[text] = {'box':box, 'conf':conf}
                
        #print ('coordinate =\n{}'.format(coordinate))
        return self.coorFilter(coordinate)

    def coorFilter(self, coordinate):
        return coordinate


class WholePage(Region):
    """When region is the whole page.

    Attributes:
        halfWidth: An integer half the width of region.

    Functions:
        coorFilter(): Filter the text by chinese numbers, and get DaTi titles.

    """
    halfWidth = 0
    def __init__(self, i, p):
        super(WholePage, self).__init__(i, p)
        self.halfWidth = int(self.width / 2)

    def coorFilter(self, coordinate):
        resultCoor = {}
        pattern = re.compile(u"一 、|二 、|三 、|四 、|五 、|六 、|七 、|八 、|九 、|十 、")

        for i,j in coordinate.items():
            num = match_dati_positive(i)
            if num != None:
                boxList = list(j['box'])
                if (boxList[0] < self.halfWidth) and (boxList[2] > self.halfWidth):
                    if pattern.match(i):
                        boxList[2] = self.halfWidth
                    else:
                        boxList[0] = self.halfWidth
                resultCoor[num] = boxList

        return resultCoor
        

class DaTiPartition(Region):
    """When region is a DaTi partition.

    Attributes:
        mark: A string marks XiaoTi.

    Functions:
        coorFilter(): Filter the text by Arabic numerals, and get XiaoTi titles.

    """
    mark = ''
    def __init__(self, i, p, m):
        super(DaTiPartition, self).__init__(i, p)
        self.mark = m

    def coorFilter(self, coordinate):
        resultCoor = {}
        #pattern = re.compile(u'(1|2|3|4|5|6|7|8|9|10|11|12|13|14|15)(\.|。|、|_|,)')
        #print ('coordinate = \n{}'.format(coordinate))
        for i,j in coordinate.items():
            num = match_xiaoti(i)
            if num != None:
                resultCoor[num] = list(j['box'])

        return resultCoor 

        
class Title():
    """Reserve the attributes of each titles

    Attributes:
        mark: A string marks title.
        nextKey: An integer key of next title.
        isMin: A boolean indicating if this title is the minimum of its region.
        isMax: A boolean indicating if this title is the maximum of its region.
        position: A array of the title's coordinate.

    """
    mark = ''
    nextKey = 0
    isMin = False
    isMax = False
    position = []
    def __init__(self, mark, nextKey, isMin, isMax, position):
        self.mark = mark
        self.nextKey = nextKey
        self.isMin = isMin
        self.isMax = isMax
        self.position = position
        

class DaTiTitle(Title):
    """When the title belongs to DaTi.
    
    Attributes:
        side: An integer indicating the title is in left side (0) or 
        right side (1) of paper.
        isMid: A boolean indicating if this title is between tow sides.

    """
    side = 0
    isMid = False
    def __init__(self, mark, side, nextKey, isMin, isMax, isMid, position):
        super(DaTiTitle, self).__init__(mark, nextKey, isMin, isMax, position)
        self.side = side
        self.isMid = isMid
        

class XiaoTiTitle(Title):
    """When the title belongs to XiaoTi."""

    def __init__(self, mark, nextKey, isMin, isMax, position):
        super(XiaoTiTitle, self).__init__(mark, nextKey, isMin, isMax, position)


def match_dati_reverse(text):
    """Search chinese numbers in text in reverse order.

    Args:
        text: A string.

    Returns:
        An integer between 0 and 10 or None.

    """
    score = "( 、|、).*(\(|〈|t|《)\d+((.(分).)|(分)|(.(分)))(\)|》)"
    if re.search(u'十'+score,text) :
        return 10
    elif re.search(u'九'+score,text) :
        return 9
    elif re.search(u'八'+score,text) :
        return 8
    elif re.search(u'七'+score,text) :
        return 7
    elif re.search(u'六'+score,text) :
        return 6
    elif re.search(u'五'+score,text) :
        return 5
    elif re.search(u'四'+score,text) :
        return 4
    elif re.search(u'三'+score,text) :
        return 3
    elif re.search(u'二'+score,text) :
        return 2
    elif re.search(u'一'+score,text) :
        return 1
    elif re.search(u'(附 加 题)|(智 慧 星)',text) :
        return 0
    else :
        return None


def match_dati_positive(text):
    """Search chinese numbers in text in positive order.

    Args:
        text: A string.

    Returns:
        An integer between 0 and 10 or None.

    """

    score = "( 、|、).*(\(|〈|t|《)\d+((.(分).)|(分)|(.(分)))(\)|》)"
    if re.search(u'一'+score,text) :
        return 1
    elif re.search(u'二'+score,text) :
        return 2
    elif re.search(u'三'+score,text) :
        return 3
    elif re.search(u'四'+score,text) :
        return 4
    elif re.search(u'五'+score,text) :
        return 5
    elif re.search(u'六'+score,text) :
        return 6
    elif re.search(u'七'+score,text) :
        return 7
    elif re.search(u'八'+score,text) :
        return 8
    elif re.search(u'九'+score,text) :
        return 9
    elif re.search(u'十'+score,text) :
        return 10
    elif re.search(u'(附 加 题)|(智 慧 星)',text) :
        return 0
    else :
        return None


def match_xiaoti(text):
    """March Arabic numerals in text.

    Args:
        text: A string.

    Returns:
        An integer between 1 and 15 or None.

    """
    dots = "(\.|。|、|_|,)"
    if re.match(u'10' + dots, text) :
        return 10
    elif re.match(u'9' + dots, text) :
        return 9
    elif re.match(u'8' + dots, text) :
        return 8
    elif re.match(u'7' + dots, text) :
        return 7
    elif re.match(u'6' + dots, text) :
        return 6
    elif re.match(u'5' + dots, text) :
        return 5
    elif re.match(u'4' + dots, text) :
        return 4
    elif re.match(u'3' + dots, text) :
        return 3
    elif re.match(u'2' + dots, text) :
        return 2
    elif re.match(u'1' + dots, text) :
        return 1
    elif re.match(u'11' + dots, text) :
        return 11
    elif re.match(u'12' + dots, text) :
        return 12
    elif re.match(u'13' + dots, text) :
        return 13
    elif re.match(u'14' + dots, text) :
        return 14
    elif re.match(u'15' + dots, text) :
        return 15
    else :
        return None
        
            
        
def template_match(imagePil, template, threshold):
    """Match template.

    Args: 
        imagePil: PIL image.
        template: Template image.
        threshold: Matching threshold.

    Returns:
        An array of four integer elements includes the coordinates 
        of the upper left corner and the lower right corner. 
        For example:

        [1,2,3,4]

    Raises: 
        ValueError: An error occurred getting the value of imagePil and template.

    """
    img_rgb = np.array(imagePil.convert('RGB'))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = np.array(template)
    coordinate = []

    w, h = template.shape[::-1]
    #h, w = template.shape[:2]  # rows->h, cols->w
    try:
        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    except Exception as e:
        print(e)
        return coordinate
    #threshold = 0.85
    #print ('threshold = {}'.format(threshold))
    #loc = np.where(res >= threshold)
    if np.max(res) >= threshold :
        loc = np.where(res == np.max(res))
        #print ('Most similar = {}'.format(np.max(res)))
        for pt in zip(*loc[::-1]):
        #cv2.rectangle(img_rgb, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), 0, 2)
        #cv2.imwrite('1-1-m.jpg', img_rgb)
        #print ('pt = {}'.format(pt))
        #print ('x1:{} y1:{} x2:{} y2:{}'.format(pt[0], pt[1], pt[0] + w, pt[1] + h))
        #if len(coordinate) == 0 :
            coordinate = [int(pt[0]), int(pt[1]), int(pt[0] + w), int(pt[1] + h)]
            #cv2.rectangle(img_rgb, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), 0, 1)
    #img_pil = Image.fromarray(img_rgb)
    #img_pil.show()
    #cv2.imwrite('1-1-m.jpg', img_rgb)
    #print ('coordinate = {}'.format(coordinate))
    return coordinate


def analyzeAnswerList(questionListDict):
    """Analyze the answer List.

    Args:
        questionListDict: A dic of question lists.

    Returns:
        questionList: A dic after analyzing, its keys is DaTi number 
        and values are numbers of XiaoTi in DaTi. For example:

        {1: 8, 2: 3}

    """
    questionList = {}
    for i in questionListDict['answerList']:
        typeOfQuestion = i['paperQuestionList'][0]['type']
        for j in i['paperQuestionList']:
            maxSort = j['sort']
            if typeOfQuestion != j['type']:
                typeOfQuestion = 70
                #break
            if maxSort < j['sort']:
                maxSort = j['sort']
        if typeOfQuestion == 11 or typeOfQuestion == 12 or maxSort >= 15:
            questionList[i['sort']] = 1
        else:
            questionList[i['sort']] = maxSort
    #print ('questionList:\n{}'.format(questionList))
    return questionList





