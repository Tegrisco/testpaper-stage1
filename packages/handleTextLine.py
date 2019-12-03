import cv2
import copy
import locale
from contextlib import contextmanager

@contextmanager
def c_locale():
    """
    c_locale()：语言设置
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


def getHandleTextLine(refImg):
    """
    getHandleTextLine(refImg):获取母卷(双栏)每一行的行高
    :param refImg: 母卷(opencv)
    :return: 母卷(双栏)每一行的行高
    """
    def leftPage(wh, img):
        """
        leftPage(wh, img):通过tesseract扫描母卷获取左栏的textline
        :param wh: 母卷的高和宽(height,width)
        :param img: 母卷(opencv)
        :return: tesseract扫描母卷左栏的textline
        """
        with c_locale():
            from tesserocr import RIL, PyTessBaseAPI, Image
            level = RIL.TEXTLINE
            img = Image.fromarray(img)
            with PyTessBaseAPI(psm=6) as api:
                api.SetImage(img)
                api.SetRectangle(0, 0, wh[1] / 2,
                                 wh[0])
                boxes = api.GetComponentImages(level, True)
                textLine = []
                print('Found {} textlines image components.'.format(len(boxes)))
                for i, (im, box, _, _) in enumerate(boxes):
                    textLine.append(box)
        return textLine

    def rightPage(wh, img):
        """
        rightPage(wh, img):通过tesseract扫描母卷获取右栏的textline
        :param wh: 母卷的高和宽(height,width)
        :param img: 母卷(opencv)
        :return: tesseract扫描母卷右栏的textline
        """
        with c_locale():
            from tesserocr import RIL, PyTessBaseAPI, Image
            level = RIL.TEXTLINE
            img = Image.fromarray(img)
            with PyTessBaseAPI(psm=6) as api:
                api.SetImage(img)
                api.SetRectangle(wh[1] / 2, 0, wh[1] / 2, wh[0])
                textLine = []
                boxes = api.GetComponentImages(level, True)
                for i, (im, box, _, _) in enumerate(boxes):
                    box['x'] = int(wh[1] / 2 + box['x'])
                    textLine.append(box)
        return textLine

    def getHandledTextLine(textLine, wh):
        """
        getHandledTextLine(textLine, wh):通过交换上一行的下边界以及下一行的上边，得到每一行的行高
        :param textLine: 通过tesseract扫描母卷获得的textline
        :param wh: 母卷的高和宽(height,width)
        :return: 每一行的最大范围{x,y,w,h}
        """
        HandledTextLine = copy.deepcopy(textLine)
        i = 0
        while i < len(HandledTextLine):
            if i == 0:
                HandledTextLine[0]['y'] = 0
                HandledTextLine[0]['h'] = textLine[i + 1]['y']
            elif i == len(HandledTextLine) - 1:
                HandledTextLine[i]['y'] = textLine[i - 1]['y'] + textLine[i - 1]['h']
                HandledTextLine[i]['h'] = wh[0]
            else:
                HandledTextLine[i]['y'] = textLine[i - 1]['y'] + textLine[i - 1]['h']
                HandledTextLine[i]['h'] = textLine[i + 1]['y'] - (textLine[i - 1]['y'] + textLine[i - 1]['h'])
            i += 1
        return HandledTextLine

    # img = cv2.imread(refImg)
    wh = refImg.shape
    handleTextLine = []
    leftTextLine = leftPage(wh, refImg)
    # print("leftTextLine:", leftTextLine)
    HandledLeftTextLine = getHandledTextLine(leftTextLine, wh)
    for leftItem in HandledLeftTextLine:
        handleTextLine.append(leftItem)
    rightTextLine = rightPage(wh, refImg)
    HandledRightTextLine = getHandledTextLine(rightTextLine, wh)
    # print('rightTextLine:', rightTextLine)
    for rightItem in HandledRightTextLine:
        handleTextLine.append(rightItem)

    return handleTextLine


if __name__ == '__main__':
    refImg = 'D://exam_img/real_image/full/14-1.jpg'
    img = cv2.imread(refImg)
    HandleTextLine = getHandleTextLine(img)
    print(HandleTextLine)
    i = 0
    for box in HandleTextLine:
        cv2.rectangle(img, (box['x'], box['y']), (box['x'] + box['w'], box['y'] + box['h']), 0, 2)
        cv2.putText(img, str(i), (box['x'], box['y']), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        i += 1
    cv2.namedWindow(refImg, cv2.WINDOW_NORMAL)
    cv2.imshow(refImg, img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # cv2.imwrite('./imgStore/textline7.jpg', img)
