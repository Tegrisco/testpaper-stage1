#作者：纪文飞 
#时间：2019-07-12
#版本：1.00

# coding=UTF-8 
# import os
# os.environ['GLOG_minloglevel'] = '3'


import numpy as np
from PIL import Image
import uuid
import sys  # coding=UTF-8
# import os
# os.environ['GLOG_minloglevel'] = '3'

import numpy as np
from PIL import Image
import uuid
import getopt
import sys
import json
import os
import time
import datetime
import urllib.request
import urllib.parse
from io import BytesIO
# import tensorflow as tf
import copy  # use deep copy
import cv2
import re
import oral1_integrate1
import gapFillAnsPos3
import preprocess
# import position
import kousuanti
import sort_position_0603
import xuehao  # 0514zzh

# import __pycache__.sigle_char_def as sr
# import __pycache__.string_char_def as ssr

num_string_model_path = '../oral/num_string_model/num_string_model'
sign_string_model_path = '../oral/sign_string_model/num_string_model'
eng_string_model_path = '../oral/eng_string_model/num_string_model'
chi_string_model_path = '../oral/chi_string_model/num_string_model'

num_dict = '../oral/dict/num_dict'
sign_dict = '../oral/dict/sign_dict'
eng_dict = '../oral/dict/eng_dict'
chi_dict = '../oral/dict/chi_dict'

running_mode = 'cpu'
save_path = '../oral'

modelPaperPath = '../oral/real_img/'
paperListPath = '../oral/ans_img/'
imgStorePath = '../oral/img_store/'
csvPath = '../oral/csv/'


def createOutJsonNoAnswerList(jsonData, coorOfPartitions, coordinateOfTitles):
    """
    无答案库创建outjson框架
    Args:
        jsonData: 这份试卷的答案库
        coorOfPartitions: 当前试卷面的大题区域坐标列表
        coordinateOfTitles:当前试卷面的大题题号区域坐标列表

    Returns: 当前试卷面的outjson

    """
    keys = list(coordinateOfTitles.keys())
    minTitle = min(keys)

    daTiNumber = len(coordinateOfTitles)
    if (str(minTitle - 1) + '-2' in coorOfPartitions):
        daTiNumber = daTiNumber + 1
        minTitle = minTitle - 1

    jsonData["studentInfo"] = {}
    jsonData["studentInfo"]["studentNo"] = ""
    jsonData["studentInfo"]["position"] = {"LX": 0, "LY": 0, "RX": 0, "RY": 0}
    jsonData["studentInfo"]["studentTempUuid"] = "uuid"
    jsonData["studentInfo"]["answerPaperIndex"] = 0
    jsonData["resultFile"] = {}
    jsonData["resultFile"]["name"] = "examName"
    jsonData["resultFile"]["size"] = []
    jsonData["resultFile"]["origPaperUrl"] = ""

    for i in range(0, daTiNumber):
        jsonData["answerList"].append({})
        jsonData["answerList"][i]["sort"] = i + minTitle
        jsonData["answerList"][i]["position"] = [{"LX": 0, "LY": 0, "RX": 0, "RY": 0}]
        jsonData["answerList"][i]["questionTypeId"] = str(uuid.uuid1())

    return jsonData


def createOutJson(jsonData, xiaoTi, coorOfPartitions, paperIdx):
    """
    有答案库创建outjson框架
    Args:
        jsonData: 这份试卷的答案库
        xiaoTi: 当前试卷面的每个中题的区域
        coorOfPartitions: 当前试卷面的大题区域
        paperIdx: 当前试卷面数（从0开始）

    Returns: 当前试卷面的outjson

    """
    paperTitleMin = min(list(xiaoTi.keys()))
    paperTitleMax = max(list(xiaoTi.keys()))

    jsonData["studentInfo"] = {}
    jsonData["studentInfo"]["studentNo"] = ""
    jsonData["studentInfo"]["position"] = {"LX": 0, "LY": 0, "RX": 0, "RY": 0}
    jsonData["studentInfo"]["studentTempUuid"] = "uuid"
    jsonData["studentInfo"]["answerPaperIndex"] = 0
    jsonData["resultFile"] = {}
    jsonData["resultFile"]["name"] = "examName"
    jsonData["resultFile"]["size"] = []
    jsonData["resultFile"]["origPaperUrl"] = ""

    for i in range(0, len(jsonData["answerList"])):
        jsonData["answerList"][i]["position"] = [{"LX": 0, "LY": 0, "RX": 0, "RY": 0}]
        # The number of answers is not based on the length of questionID in input json data
        for j in range(0, len(jsonData["answerList"][i]["paperQuestionList"])):
            jsonData["answerList"][i]["paperQuestionList"][j]["position"] = [{"LX": 0, "LY": 0, "RX": 0, "RY": 0}]
            jsonData["answerList"][i]["paperQuestionList"][j]["result"] = []
            for k in range(0, len(jsonData["answerList"][i]["paperQuestionList"][j]["answer"])):
                jsonData["answerList"][i]["paperQuestionList"][j]["result"].append({"resultFlag": False, \
                                                                                    "position": [
                                                                                        {"LX": 0, "LY": 0, "RX": 0,
                                                                                         "RY": 0}], \
                                                                                    "recognitionResult": ""})
    # Delete daTi that are not on the same side
    for i in range(0, len(jsonData["answerList"])):
        daTiNum = i + 1  # daTiNum starts from 1 instead of zero
        if (daTiNum in xiaoTi):
            continue
        else:
            for j in range(0, len(jsonData['answerList'])):
                if (jsonData['answerList'][j]['sort'] == daTiNum):
                    del jsonData['answerList'][j]
                    break  # if found, directly exit from current loop
    # When the topic is across, delete the topic that is not on the same side.
    # Second paper
    if (paperIdx == 1) and \
            (str(paperTitleMin - 1) + '-2' not in coorOfPartitions):
        zhongTiNum = len(xiaoTi[paperTitleMin])
        totalNumOfZhongTi = len(jsonData['answerList'][0]['paperQuestionList'])
        for i in range(0, totalNumOfZhongTi):
            zhongTi = i + 1  # zhongTiNum starts from 1 instead of zero
            if (zhongTi > totalNumOfZhongTi - zhongTiNum):
                continue
            else:
                for j in range(0, len(jsonData['answerList'][0]['paperQuestionList'])):
                    if (jsonData['answerList'][0]['paperQuestionList'][j]['sort'] <= zhongTi):
                        del jsonData['answerList'][0]['paperQuestionList'][j]
                        break
    # First paper
    elif (paperIdx == 0) and \
            (jsonData['answerList'][paperTitleMax - paperTitleMin]['paperQuestionList'][0]['type'] != 11):
        zhongTiNum = len(xiaoTi[paperTitleMax])
        totalNumOfZhongTi = len(jsonData['answerList'][paperTitleMax - paperTitleMin]['paperQuestionList'])
        for i in range(0, totalNumOfZhongTi):
            zhongTi = i + 1  # zhongTiNum starts from 1 instead of zero
            if (zhongTi <= zhongTiNum):
                continue
            else:
                for j in range(0, len(jsonData['answerList'][paperTitleMax - paperTitleMin]['paperQuestionList'])):
                    if (jsonData['answerList'][paperTitleMax - paperTitleMin]['paperQuestionList'][j][
                        'sort'] >= zhongTi):
                        del jsonData['answerList'][paperTitleMax - paperTitleMin]['paperQuestionList'][j]
                        break
    return jsonData


def screenshot(outJson, ansImg, daTiNum, titleMin, oralArithmeticCoor, completionPosition, choicePosition,
               judgementPosition):
    """
    截取答案区域的截图，并以对应的类型和所在题数命名保存到对应的图片路径
    Args:
        outJson: 当前试卷面的outjson
        ansImg: 分离卷
        daTiNum: 当前试卷面目前的大题序号
        titleMin: 当前试卷面目前的大题sort
        oralArithmeticCoor: 口算题的答案+题目坐标区域
        completionPosition: 填空题中每个答案区域的坐标
        choicePosition: 选择题中每个答案区域的坐标
        judgementPosition: 判断题中每个答案区域的坐标

    Returns: 无

    """
    for j in range(0, len(outJson['answerList'][daTiNum]['paperQuestionList'])):
        middleNum = j + outJson['answerList'][daTiNum]['paperQuestionList'][0]['sort']
        xiTiAnswer = outJson['answerList'][daTiNum]['paperQuestionList'][j]['answer']
        if (len(oralArithmeticCoor) > 0):
            if (len(oralArithmeticCoor[0]) > j):
                if (len(re.findall(r"[0-9]", xiTiAnswer[0])) == 1) and \
                        (oralArithmeticCoor[4][j] - oralArithmeticCoor[2][j] > 2) and \
                        (oralArithmeticCoor[3][j] - oralArithmeticCoor[1][j] > 2):
                    kouSuanCropImg = ansImg[oralArithmeticCoor[2][j]:oralArithmeticCoor[4][j], \
                                     oralArithmeticCoor[1][j]:oralArithmeticCoor[3][j]]
                    cv2.imwrite(num_img_path + "/num_" + str(daTiNum + titleMin) + "." + str(j + 1) + ".jpg",
                                kouSuanCropImg)
                elif (len(re.findall(r"[0-9]", xiTiAnswer[0])) > 1) and \
                        (oralArithmeticCoor[4][j] - oralArithmeticCoor[2][j] > 2) and \
                        (oralArithmeticCoor[3][j] - oralArithmeticCoor[1][j] > 2):
                    kouSuanCropImg = ansImg[oralArithmeticCoor[2][j]:oralArithmeticCoor[4][j], \
                                     oralArithmeticCoor[1][j]:oralArithmeticCoor[3][j]]
                    cv2.imwrite(num_string_img_path + "/num_" + str(daTiNum + titleMin) + "." + str(j + 1) + ".jpg",
                                kouSuanCropImg)
        elif (middleNum in completionPosition):
            if (len(completionPosition[middleNum]) > 0):
                # Number of the xiaoti answer
                answerPositionNum = len(xiTiAnswer)
                if (len(xiTiAnswer) > len(completionPosition[middleNum])):
                    answerPositionNum = len(completionPosition[middleNum])
                for k in range(0, answerPositionNum):
                    if (completionPosition[middleNum][k][3] - completionPosition[middleNum][k][1] > 2) and \
                            (completionPosition[middleNum][k][2] - completionPosition[middleNum][k][0] > 2):
                        if (len(re.findall(r"[0-9]", xiTiAnswer[k])) == 1) and (len(xiTiAnswer[k]) == 1):
                            tianKongCropImg = ansImg[
                                              completionPosition[middleNum][k][1]:completionPosition[middleNum][k][3], \
                                              completionPosition[middleNum][k][0]:completionPosition[middleNum][k][2]]
                            cv2.imwrite(
                                num_img_path + "/num_" + str(daTiNum + titleMin) + "." + str(middleNum) + "." + str(
                                    k + 1) + ".jpg", tianKongCropImg)
                        elif (len(re.findall(r"[0-9]", xiTiAnswer[k])) > 1) and (
                                len(re.findall(r"[0-9]", xiTiAnswer[k])) == len(xiTiAnswer[k])):
                            tianKongCropImg = ansImg[
                                              completionPosition[middleNum][k][1]:completionPosition[middleNum][k][3], \
                                              completionPosition[middleNum][k][0]:completionPosition[middleNum][k][2]]
                            cv2.imwrite(num_string_img_path + "/num_" + str(daTiNum + titleMin) + "." + str(
                                middleNum) + "." + str(k + 1) + ".jpg", tianKongCropImg)
                        elif (len(re.findall(r"[\u4e00-\u9fa5]", xiTiAnswer[k])) == 1) and (len(xiTiAnswer[k]) == 1):
                            tianKongCropImg = ansImg[
                                              completionPosition[middleNum][k][1]:completionPosition[middleNum][k][3], \
                                              completionPosition[middleNum][k][0]:completionPosition[middleNum][k][2]]
                            cv2.imwrite(
                                chi_img_path + "/chi_" + str(daTiNum + titleMin) + "." + str(middleNum) + "." + str(
                                    k + 1) + ".jpg", tianKongCropImg)
                        elif (len(re.findall(r"[\u4e00-\u9fa5]", xiTiAnswer[k])) > 1) and \
                                (len(re.findall(r"[\u4e00-\u9fa5]", xiTiAnswer[k])) == len(xiTiAnswer[k])):
                            tianKongCropImg = ansImg[
                                              completionPosition[middleNum][k][1]:completionPosition[middleNum][k][3], \
                                              completionPosition[middleNum][k][0]:completionPosition[middleNum][k][2]]
                            cv2.imwrite(chi_string_img_path + "/chi_" + str(daTiNum + titleMin) + "." + str(
                                middleNum) + "." + str(k + 1) + ".jpg", tianKongCropImg)
                        elif (len(re.findall(r"[\u0041-\u005a,\u0061-\u007a]", xiTiAnswer[k])) == 1) and (
                                len(xiTiAnswer[k]) == 1):
                            tianKongCropImg = ansImg[
                                              completionPosition[middleNum][k][1]:completionPosition[middleNum][k][3], \
                                              completionPosition[middleNum][k][0]:completionPosition[middleNum][k][2]]
                            cv2.imwrite(
                                eng_img_path + "/eng_" + str(daTiNum + titleMin) + "." + str(middleNum) + "." + str(
                                    k + 1) + ".jpg", tianKongCropImg)
                        elif (len(re.findall(r"[\u0041-\u005a,\u0061-\u007a]", xiTiAnswer[k])) > 1) and \
                                (len(re.findall(r"[\u0041-\u005a,\u0061-\u007a]", xiTiAnswer[k])) == len(
                                    xiTiAnswer[k])):
                            tianKongCropImg = ansImg[
                                              completionPosition[middleNum][k][1]:completionPosition[middleNum][k][3], \
                                              completionPosition[middleNum][k][0]:completionPosition[middleNum][k][2]]
                            cv2.imwrite(eng_string_img_path + "/eng_" + str(daTiNum + titleMin) + "." + str(
                                middleNum) + "." + str(k + 1) + ".jpg", tianKongCropImg)
                        elif (len(xiTiAnswer[k]) > 3):
                            tianKongCropImg = ansImg[
                                              completionPosition[middleNum][k][1]:completionPosition[middleNum][k][3], \
                                              completionPosition[middleNum][k][0]:completionPosition[middleNum][k][2]]
                            cv2.imwrite(sign_string_img_path + "/sign_" + str(daTiNum + titleMin) + "." + str(
                                middleNum) + "." + str(k + 1) + ".jpg", tianKongCropImg)
                        else:
                            tianKongCropImg = ansImg[
                                              completionPosition[middleNum][k][1]:completionPosition[middleNum][k][3], \
                                              completionPosition[middleNum][k][0]:completionPosition[middleNum][k][2]]
                            cv2.imwrite(
                                sign_img_path + "/sign_" + str(daTiNum + titleMin) + "." + str(middleNum) + "." + str(
                                    k + 1) + ".jpg", tianKongCropImg)

        elif (middleNum in choicePosition):
            if (len(choicePosition[middleNum]) > 0):
                # Number of the xiaoti answer
                answerPositionNum = len(xiTiAnswer)
                if (len(xiTiAnswer) > len(choicePosition[middleNum])):
                    answerPositionNum = len(choicePosition[middleNum])
                for k in range(0, answerPositionNum):
                    if (choicePosition[middleNum][k][3] - choicePosition[middleNum][k][1] > 2) and \
                            (choicePosition[middleNum][k][2] - choicePosition[middleNum][k][0] > 2):
                        if (len(re.findall(r"[\u0041-\u005a,\u0061-\u007a]", xiTiAnswer[k])) == 1) and (
                                len(xiTiAnswer[k]) == 1):
                            choiceCropImg = ansImg[choicePosition[middleNum][k][1]:choicePosition[middleNum][k][3], \
                                            choicePosition[middleNum][k][0]:choicePosition[middleNum][k][2]]
                            cv2.imwrite(
                                eng_img_path + "/eng_" + str(daTiNum + titleMin) + "." + str(middleNum) + "." + str(
                                    k + 1) + ".jpg", choiceCropImg)
                        elif (len(re.findall(r"[\u0041-\u005a,\u0061-\u007a]", xiTiAnswer[k])) > 1) and \
                                (len(re.findall(r"[\u0041-\u005a,\u0061-\u007a]", xiTiAnswer[k])) == len(
                                    xiTiAnswer[k])):
                            choiceCropImg = ansImg[choicePosition[middleNum][k][1]:choicePosition[middleNum][k][3], \
                                            choicePosition[middleNum][k][0]:choicePosition[middleNum][k][2]]
                            cv2.imwrite(eng_string_img_path + "/eng_" + str(daTiNum + titleMin) + "." + str(
                                middleNum) + "." + str(k + 1) + ".jpg", choiceCropImg)
                        elif (len(xiTiAnswer[k]) > 3):
                            choiceCropImg = ansImg[choicePosition[middleNum][k][1]:choicePosition[middleNum][k][3], \
                                            choicePosition[middleNum][k][0]:choicePosition[middleNum][k][2]]
                            cv2.imwrite(sign_string_img_path + "/sign_" + str(daTiNum + titleMin) + "." + str(
                                middleNum) + "." + str(k + 1) + ".jpg", choiceCropImg)
                        else:
                            choiceCropImg = ansImg[choicePosition[middleNum][k][1]:choicePosition[middleNum][k][3], \
                                            choicePosition[middleNum][k][0]:choicePosition[middleNum][k][2]]
                            cv2.imwrite(
                                sign_img_path + "/sign_" + str(daTiNum + titleMin) + "." + str(middleNum) + "." + str(
                                    k + 1) + ".jpg", choiceCropImg)
        elif (middleNum in judgementPosition):
            if (len(judgementPosition[middleNum]) > 0):
                # Number of the xiaoti answer
                answerPositionNum = len(xiTiAnswer)
                if (len(xiTiAnswer) > len(judgementPosition[middleNum])):
                    answerPositionNum = len(judgementPosition[middleNum])
                for k in range(0, answerPositionNum):
                    if (judgementPosition[middleNum][k][3] - judgementPosition[middleNum][k][1] > 2) and \
                            (judgementPosition[middleNum][k][2] - judgementPosition[middleNum][k][0] > 2):
                        judgementCropImg = ansImg[judgementPosition[middleNum][k][1]:judgementPosition[middleNum][k][3], \
                                           judgementPosition[middleNum][k][0]:judgementPosition[middleNum][k][2]]
                        cv2.imwrite(
                            sign_img_path + "/sign_" + str(daTiNum + titleMin) + "." + str(middleNum) + "." + str(
                                k + 1) + ".jpg", judgementCropImg)


def analyzerResult():
    """
    调用图片识别模块对截图进行识别，将识别的结果以数组形式输出
    Returns: numResult, numStringResult, signResult, engResult, chiResult

    """
    try:
        numResult = []
        fNum = open(csvPath + timeN + 'num_ans_result.csv', "r", encoding="utf-8")
        for line in fNum:
            line = line.replace("\n", "")
            numResult.append(line.split(","))
        fNum.close()
    except IOError:
        print('No such file or directory ...num_ans_result.csv', timeN)

    try:
        numStringResult = []
        fNumString = open(csvPath + timeN + 'num_string_ans_result.csv', "r", encoding="utf-8")
        for line in fNumString:
            line = line.replace("\n", "")
            numStringResult.append(line.split(","))
        fNumString.close()
    except IOError:
        print('No such file or directory ...num_string_ans_result.csv', timeN)

    try:
        signResult = []
        fSign = open(csvPath + timeN + 'sign_ans_result.csv', "r", encoding="utf-8")
        for line in fSign:
            line = line.replace("\n", "")
            signResult.append(line.split(","))
        fSign.close()
    except IOError:
        print('No such file or directory ...sign_ans_result.csv', timeN)

    # try:
    # signStringResult = []
    # fSignString = open('./csv/'+timeN+'sign_string_ans_result.csv',"r",encoding = "utf-8")
    # for line in fSignString:
    # line = line.replace("\n","")
    # signStringResult.append(line.split(","))
    # fSignString.close()
    # except IOError:
    # print ('No such file or directory ...sign_string_ans_result.csv', timeN)

    try:
        engResult = []
        fEng = open(csvPath + timeN + 'eng_ans_result.csv', "r", encoding="utf-8")
        for line in fEng:
            line = line.replace("\n", "")
            engResult.append(line.split(","))
        fEng.close()
    except IOError:
        print('No such file or directory ...eng_ans_result.csv', timeN)

    # try:
    # engStringResult = []
    # fEngString = open('./csv/'+timeN+'eng_string_ans_result.csv',"r",encoding = "utf-8")
    # for line in fEngString:
    # line = line.replace("\n","")
    # engStringResult.append(line.split(","))
    # fEngString.close()
    # except IOError:
    # print ('No such file or directory ...eng_string_ans_result.csv', timeN)

    try:
        chiResult = []
        fChi = open(csvPath + timeN + 'chi_ans_result.csv', "r", encoding="utf-8")
        for line in fChi:
            line = line.replace("\n", "")
            chiResult.append(line.split(","))
        fChi.close()
    except IOError:
        print('No such file or directory ...chi_ans_result.csv', timeN)

    # try:
    # chiStringResult = []
    # fChiString = open('./csv/'+timeN+'chi_string_ans_result.csv',"r",encoding = "utf-8")
    # for line in fChiString:
    # line = line.replace("\n","")
    # chiStringResult.append(line.split(","))
    # fChiString.close()
    # except IOError:
    # print ('No such file or directory ...chi_string_ans_result ', FileNotFoundError)

    return numResult, numStringResult, signResult, engResult, chiResult


def daTiCoordinateW(outJson, titleMin, paperIdx, coorOfPartitions, coordinateOfTitles, completeJsonData):
    """
    将当前试卷面的大题区域坐标写入outjson
    Args:
        outJson: 当前试卷面的outjson
        titleMin: 当前试卷面目前大题的sort
        paperIdx: 当前试卷面数（从0开始）
        coorOfPartitions: 当前试卷面大题区域坐标
        coordinateOfTitles: 当前试卷面大题题干坐标
        completeJsonData: 到目前为止所有试卷面的outjson数组集

    Returns: 无

    """
    for i in range(0, len(outJson['answerList'])):
        if (str(i + titleMin) in coorOfPartitions):
            outJson['answerList'][i]["position"][0]['LX'] = coordinateOfTitles[i + titleMin][0]
            outJson['answerList'][i]["position"][0]['LY'] = coorOfPartitions[str(i + titleMin)][1]
            outJson['answerList'][i]["position"][0]['RX'] = coorOfPartitions[str(i + titleMin)][2]
            outJson['answerList'][i]["position"][0]['RY'] = coorOfPartitions[str(i + titleMin)][3]
        elif (str(i + titleMin) + '-1' in coorOfPartitions):
            outJson['answerList'][i]["position"][0]['LX'] = coordinateOfTitles[i + titleMin][0]
            outJson['answerList'][i]["position"][0]['LY'] = coorOfPartitions[str(i + titleMin) + '-1'][1]
            outJson['answerList'][i]["position"][0]['RX'] = coorOfPartitions[str(i + titleMin) + '-1'][2]
            outJson['answerList'][i]["position"][0]['RY'] = coorOfPartitions[str(i + titleMin) + '-1'][3]
            if (str(i + titleMin) + '-2' in coorOfPartitions):
                outJson['answerList'][i]["position"].append({"LX": coorOfPartitions[str(i + titleMin) + '-2'][0], \
                                                             "LY": coorOfPartitions[str(i + titleMin) + '-2'][1], \
                                                             "RX": coorOfPartitions[str(i + titleMin) + '-2'][2], \
                                                             "RY": coorOfPartitions[str(i + titleMin) + '-2'][3]})
        elif (str(i + titleMin) + '-2' in coorOfPartitions):
            outJson['answerList'][i]["position"][0]['LX'] = coorOfPartitions[str(i + titleMin) + '-2'][1]
            outJson['answerList'][i]["position"][0]['LY'] = coorOfPartitions[str(i + titleMin) + '-2'][1]
            outJson['answerList'][i]["position"][0]['RX'] = coorOfPartitions[str(i + titleMin) + '-2'][2]
            outJson['answerList'][i]["position"][0]['RY'] = coorOfPartitions[str(i + titleMin) + '-2'][3]
            if (paperIdx == 1):
                outJson["answerList"][i]["questionTypeId"] = completeJsonData[-1]["answerList"][-1]["questionTypeId"]


def zhongTiPositionW(outJson, xiaoTi):
    """
    将当前试卷面的中题区域坐标写入outjson
    Args:
        outJson: 当前试卷面的outjson
        xiaoTi: 当前试卷面的中题区域的坐标

    Returns:

    """
    titleBegin = outJson['answerList'][0]['sort']
    for i in range(0, len(outJson['answerList'])):
        if (len(outJson['answerList'][i]['paperQuestionList']) <= len(xiaoTi[i + titleBegin])):
            zhongTiBeginNum = outJson['answerList'][i]['paperQuestionList'][0]['sort']
            if (len(xiaoTi[i + titleBegin]) == 1):
                zhongTiBeginNum = 1
        if (len(outJson['answerList'][i]['paperQuestionList']) == 1):
            for single in xiaoTi[i + titleBegin]:
                outJson['answerList'][i]['paperQuestionList'][0]['position'][0]['LX'] = xiaoTi[i + titleBegin][single][
                    0]
                outJson['answerList'][i]['paperQuestionList'][0]['position'][0]['LY'] = xiaoTi[i + titleBegin][single][
                    1]
                outJson['answerList'][i]['paperQuestionList'][0]['position'][0]['RX'] = xiaoTi[i + titleBegin][single][
                    2]
                outJson['answerList'][i]['paperQuestionList'][0]['position'][0]['RY'] = xiaoTi[i + titleBegin][single][
                    3]
                if (len(xiaoTi[i + titleBegin][single]) == 8):
                    outJson['answerList'][i]['paperQuestionList'][0]['position'].append(
                        {"LX": xiaoTi[i + titleBegin][single][4], \
                         "LY": xiaoTi[i + titleBegin][single][5], \
                         "RX": xiaoTi[i + titleBegin][single][6], \
                         "RY": xiaoTi[i + titleBegin][single][7]})
        else:
            for j in range(0, len(outJson['answerList'][i]['paperQuestionList'])):
                zhongTiSort = outJson['answerList'][i]['paperQuestionList'][j]['sort']
                if (zhongTiSort in xiaoTi[i + titleBegin]):
                    outJson['answerList'][i]['paperQuestionList'][j]['position'][0]['LX'] = \
                        xiaoTi[i + titleBegin][zhongTiSort][0]
                    outJson['answerList'][i]['paperQuestionList'][j]['position'][0]['LY'] = \
                        xiaoTi[i + titleBegin][zhongTiSort][1]
                    outJson['answerList'][i]['paperQuestionList'][j]['position'][0]['RX'] = \
                        xiaoTi[i + titleBegin][zhongTiSort][2]
                    outJson['answerList'][i]['paperQuestionList'][j]['position'][0]['RY'] = \
                        xiaoTi[i + titleBegin][zhongTiSort][3]
                    if (len(xiaoTi[i + titleBegin][zhongTiSort]) == 8):
                        outJson['answerList'][i]['paperQuestionList'][j]['position'].append(
                            {"LX": xiaoTi[i + titleBegin][zhongTiSort][4], \
                             "LY": xiaoTi[i + titleBegin][zhongTiSort][5], \
                             "RX": xiaoTi[i + titleBegin][zhongTiSort][6], \
                             "RY": xiaoTi[i + titleBegin][zhongTiSort][7]})


def oralArithmeticAnswerL(outJson, daTiNum, titleMin, oralArithmeticCoor, numResult, numStringResult):
    """
    将当前试卷面中口算题大题的题目和答案区域的坐标和答案的正确与否写入对应的outjson位置中
    Args:
        outJson: 当前试卷面的outjson
        daTiNum: 当前试卷面目前口算题的序号
        titleMin: 当前试卷面目前口算题的sort
        oralArithmeticCoor: 当前试卷面口算题中的各个题目和答案区域的坐标
        numResult: 当前试卷面数字单字符的识别结果
        numStringResult: 当前试卷面数字多字符的识别结果

    Returns: 无

    """
    kouSuanNum = len(outJson['answerList'][daTiNum]['paperQuestionList'])
    if (kouSuanNum > len(oralArithmeticCoor[0])):
        kouSuanNum = len(oralArithmeticCoor[0])
    for i in range(0, kouSuanNum):
        outJson['answerList'][daTiNum]['paperQuestionList'][i]['position'][0]['LX'] = oralArithmeticCoor[0][i]
        outJson['answerList'][daTiNum]['paperQuestionList'][i]['position'][0]['LY'] = oralArithmeticCoor[2][i]
        outJson['answerList'][daTiNum]['paperQuestionList'][i]['position'][0]['RX'] = oralArithmeticCoor[3][i]
        outJson['answerList'][daTiNum]['paperQuestionList'][i]['position'][0]['RY'] = oralArithmeticCoor[4][i]
        outJson['answerList'][daTiNum]['paperQuestionList'][i]['result'][0]['position'][0]['LX'] = \
            oralArithmeticCoor[1][i]
        outJson['answerList'][daTiNum]['paperQuestionList'][i]['result'][0]['position'][0]['LY'] = \
            oralArithmeticCoor[2][i]
        outJson['answerList'][daTiNum]['paperQuestionList'][i]['result'][0]['position'][0]['RX'] = \
            oralArithmeticCoor[3][i]
        outJson['answerList'][daTiNum]['paperQuestionList'][i]['result'][0]['position'][0]['RY'] = \
            oralArithmeticCoor[4][i]
        for j in range(1, len(numResult)):
            if (numResult[j][0] == str(daTiNum + titleMin) + '.' + str(i + 1)):
                outJson['answerList'][daTiNum]['paperQuestionList'][i]['result'][0]['recognitionResult'] = numResult[j][
                    1]
                if (numResult[j][1] == outJson['answerList'][daTiNum]['paperQuestionList'][i]['answer'][0]):
                    outJson['answerList'][daTiNum]['paperQuestionList'][i]['result'][0]['resultFlag'] = True
        for j in range(1, len(numStringResult)):
            if (numStringResult[j][0] == str(daTiNum + titleMin) + '.' + str(i + 1)):
                outJson['answerList'][daTiNum]['paperQuestionList'][i]['result'][0]['recognitionResult'] = \
                    numStringResult[j][1]
                if (numStringResult[j][1] == outJson['answerList'][daTiNum]['paperQuestionList'][i]['answer'][0]):
                    outJson['answerList'][daTiNum]['paperQuestionList'][i]['result'][0]['resultFlag'] = True


def completionAnswerL(outJson, daTiNum, titleMin, completionMiddle, numResult, numStringResult, signResult, engResult,
                      chiResult):
    """
    将当前试卷面中填空题大题的题目和答案区域的坐标和答案的正确与否写入对应的outjson位置中
    Args:
        outJson: 当前试卷面的outjson
        daTiNum: 当前那试卷面目前的填空题序号
        titleMin: 当前那试卷面目前的填空题sort
        completionMiddle: 当前那试卷面目前的填空题中各个中题的区域坐标
        numResult: 当前试卷面数字单字符识别结果
        numStringResult: 当前试卷面数字多字符识别结果
        signResult: 当前试卷面特殊字符单字符识别结果
        engResult: 当前试卷面英文单字符识别结果
        chiResult: 当前试卷面中文单字符识别结果

    Returns: 无

    """
    for i in completionMiddle:
        if (len(completionMiddle[i]) > 0):
            completionI = i
            if (outJson['answerList'][daTiNum]['paperQuestionList'][0]['sort'] != 1):
                i = i - outJson['answerList'][daTiNum]['paperQuestionList'][0]['sort'] + 1
            tianKongAnswerNum = len(outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['answer'])
            if (len(outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['answer']) > len(
                    completionMiddle[completionI])):
                tianKongAnswerNum = len(completionMiddle[completionI])
            for j in range(0, tianKongAnswerNum):
                outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j]['position'][0]['LX'] = \
                    completionMiddle[completionI][j][0]
                outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j]['position'][0]['LY'] = \
                    completionMiddle[completionI][j][1]
                outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j]['position'][0]['RX'] = \
                    completionMiddle[completionI][j][2]
                outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j]['position'][0]['RY'] = \
                    completionMiddle[completionI][j][3]
                if (outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['answer'][j] == "None") and \
                        (len(outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j]['position'][
                                 0]) == 4):
                    outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j]['resultFlag'] = True

                tianKongAnswer = outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['answer'][j]

                if (len(re.findall(r"[0-9]", tianKongAnswer)) == 1) and (len(tianKongAnswer) == 1):
                    for k in range(1, len(numResult)):
                        if (numResult[k][0] == str(daTiNum + titleMin) + '.' + str(completionI) + '.' + str(j + 1)):
                            outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j][
                                'recognitionResult'] = numResult[k][1]
                            if (numResult[k][1] == tianKongAnswer):
                                outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j][
                                    'resultFlag'] = True
                elif (len(re.findall(r"[0-9]", tianKongAnswer)) > 1) and (
                        len(re.findall(r"[0-9]", tianKongAnswer)) == len(tianKongAnswer)):
                    for k in range(1, len(numStringResult)):
                        if (numStringResult[k][0] == str(daTiNum + titleMin) + '.' + str(completionI) + '.' + str(
                                j + 1)):
                            outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j][
                                'recognitionResult'] = numStringResult[k][1]
                            if (numStringResult[k][1] == tianKongAnswer):
                                outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j][
                                    'resultFlag'] = True
                elif (len(re.findall(r"[\u4e00-\u9fa5]", tianKongAnswer)) == 1) and (len(tianKongAnswer) == 1):
                    for k in range(1, len(chiResult)):
                        if (chiResult[k][0] == str(daTiNum + titleMin) + '.' + str(completionI) + '.' + str(j + 1)):
                            outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j][
                                'recognitionResult'] = chiResult[k][1]
                            if (chiResult[k][1] == tianKongAnswer):
                                outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j][
                                    'resultFlag'] = True
                elif (len(re.findall(r"[\u0041-\u005a,\u0061-\u007a]", tianKongAnswer)) == 1) and (
                        len(tianKongAnswer) == 1):
                    for k in range(1, len(engResult)):
                        if (engResult[k][0] == str(daTiNum + titleMin) + '.' + str(completionI) + '.' + str(j + 1)):
                            outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j][
                                'recognitionResult'] = engResult[k][1]
                            if (engResult[k][1] == tianKongAnswer):
                                outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j][
                                    'resultFlag'] = True
                # elif (len(re.findall(r"[\u0041-\u005a,\u0061-\u007a]", tianKongAnswer)) > 1) and \
                # (len(re.findall(r"[\u0041-\u005a,\u0061-\u007a]", tianKongAnswer)) == len(tianKongAnswer)):
                # for k in range(1, len(engStringResult)):
                # if (engStringResult[k][0] == str(daTiNum+titleMin)+'.'+str(i)+'.'+str(j+1)):
                # outJson['answerList'][daTiNum]['paperQuestionList'][i-1]['result'][j]['recognitionResult'] = engStringResult[k][1]
                # if (engStringResult[k][1] == tianKongAnswer):
                # outJson['answerList'][daTiNum]['paperQuestionList'][i-1]['result'][j]['resultFlag'] = True
                else:
                    for k in range(1, len(signResult)):
                        if (signResult[k][0] == str(daTiNum + titleMin) + '.' + str(completionI) + '.' + str(j + 1)):
                            outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j][
                                'recognitionResult'] = signResult[k][1]
                            # print('outJsonAnswer....', tianKongAnswer)
                            # print(
                            #     'dati :{} zhonti :{} xiaoti :{} signResult: {}'.format(daTiNum + titleMin, completionI,
                            #                                                            j + 1, signResult[k][1]))
                            if (signResult[k][1] == tianKongAnswer):
                                outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j][
                                    'resultFlag'] = True
                            elif (signResult[k][1] == "False") and (tianKongAnswer == "×"):
                                outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j][
                                    'resultFlag'] = True


def choiceAnswerL(outJson, daTiNum, titleMin, choiceMiddle, signResult, engResult):
    """
    将当前试卷面中选择题大题的题目和答案区域的坐标和答案的正确与否写入对应的outjson位置中
    Args:
        outJson: 当前试卷面的outjson
        daTiNum: 当前那试卷面目前的选择题序号
        titleMin: 当前那试卷面目前的选择题sort
        choiceMiddle: 当前那试卷面目前的选择题中各个中题的区域坐标
        signResult: 当前试卷面特殊字符单字符识别结果
        engResult: 当前试卷面英文单字符识别结果

    Returns: 无

    """
    for i in choiceMiddle:
        if (len(choiceMiddle[i]) > 0):
            choiceI = i
            if (outJson['answerList'][daTiNum]['paperQuestionList'][0]['sort'] != 1):
                i = i - outJson['answerList'][daTiNum]['paperQuestionList'][0]['sort'] + 1
            choiceAnswerNum = len(outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['answer'])
            if (len(outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['answer']) > len(choiceMiddle[choiceI])):
                choiceAnswerNum = len(choiceMiddle[choiceI])
            for j in range(0, choiceAnswerNum):
                # print ('j.....', j)
                outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j]['position'][0]['LX'] = \
                    choiceMiddle[choiceI][j][0]
                outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j]['position'][0]['LY'] = \
                    choiceMiddle[choiceI][j][1]
                outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j]['position'][0]['RX'] = \
                    choiceMiddle[choiceI][j][2]
                outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j]['position'][0]['RY'] = \
                    choiceMiddle[choiceI][j][3]
                for k in range(1, len(signResult)):
                    if (signResult[k][0] == str(daTiNum + titleMin) + '.' + str(choiceI) + '.' + str(j + 1)):
                        outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j]['recognitionResult'] = \
                            signResult[k][1]
                        if (signResult[k][1] == outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['answer'][
                            j]):
                            outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j]['resultFlag'] = True
                for k in range(1, len(engResult)):
                    if (engResult[k][0] == str(daTiNum + titleMin) + '.' + str(choiceI) + '.' + str(j + 1)):
                        outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j]['recognitionResult'] = \
                            engResult[k][1]
                        if (engResult[k][1] == outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['answer'][j]):
                            outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j]['resultFlag'] = True


def judgementAnswerL(outJson, daTiNum, titleMin, judgementMiddle, signResult):
    """
    将当前试卷面中判断题大题的题目和答案区域的坐标和答案的正确与否写入对应的outjson位置中
    Args:
        outJson: 当前试卷面的outjson
        daTiNum: 当前那试卷面目前的判断题序号
        titleMin: 当前那试卷面目前的判断题sort
        judgementMiddle: 当前那试卷面目前的判断题中各个中题的区域坐标
        signResult: 当前试卷面特殊字符单字符识别结果

    Returns: 无

    """
    for i in judgementMiddle:
        if (len(judgementMiddle[i]) > 0):
            judgementI = i
            if (outJson['answerList'][daTiNum]['paperQuestionList'][0]['sort'] != 1):
                i = i - outJson['answerList'][daTiNum]['paperQuestionList'][0]['sort'] + 1
            judgementAnswerNum = len(outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['answer'])
            if (len(outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['answer']) > len(
                    judgementMiddle[judgementI])):
                judgementAnswerNum = len(judgementMiddle[judgementI])
            for j in range(0, judgementAnswerNum):
                outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j]['position'][0]['LX'] = \
                    judgementMiddle[judgementI][j][0]
                outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j]['position'][0]['LY'] = \
                    judgementMiddle[judgementI][j][1]
                outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j]['position'][0]['RX'] = \
                    judgementMiddle[judgementI][j][2]
                outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j]['position'][0]['RY'] = \
                    judgementMiddle[judgementI][j][3]
                for k in range(1, len(signResult)):
                    if (signResult[k][0] == str(daTiNum + titleMin) + '.' + str(judgementI) + '.' + str(j + 1)):
                        outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j]['recognitionResult'] = \
                            signResult[k][1]
                        if (signResult[k][1] == outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['answer'][
                            j]):
                            outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j]['resultFlag'] = True
                        elif (signResult[k][1] == "False"):
                            outJson['answerList'][daTiNum]['paperQuestionList'][i - 1]['result'][j]['resultFlag'] = True


def screenshotDeepL(deep, outJson, ansImg, daTiNum, titleMin, oralArithmeticCoor, completionPosition,
                    choicePosition, judgementPosition):
    """
    将当前测试的所有试卷面的答案区域截图都存放指定路径中并按数字递增命名
    Args:
        deep: 存放截图文件的路径
        outJson: 当前试卷面的outjson
        ansImg: 分离卷
        daTiNum: 当前那试卷面目前的大题序号
        titleMin: 当前那试卷面目前的大题sort
        oralArithmeticCoor: 口算题的答案+题目坐标区域
        completionPosition: 填空题中每个答案区域的坐标
        choicePosition: 选择题中每个答案区域的坐标
        judgementPosition: 判断题中每个答案区域的坐标

    Returns: 无

    """
    for j in range(0, len(outJson['answerList'][daTiNum]['paperQuestionList'])):
        middleNum = j + outJson['answerList'][daTiNum]['paperQuestionList'][0]['sort']
        xiTiAnswer = outJson['answerList'][daTiNum]['paperQuestionList'][j]['answer']
        if (len(oralArithmeticCoor) > 0):
            if (len(oralArithmeticCoor[0]) > j):
                if (len(re.findall(r"[0-9]", xiTiAnswer[0])) == 1) and \
                        (oralArithmeticCoor[4][j] - oralArithmeticCoor[2][j] > 2) and \
                        (oralArithmeticCoor[3][j] - oralArithmeticCoor[1][j] > 2):
                    kouSuanCropImg = ansImg[oralArithmeticCoor[2][j]:oralArithmeticCoor[4][j], \
                                     oralArithmeticCoor[1][j]:oralArithmeticCoor[3][j]]

                    if not os.path.exists(deep + '/numImg/' + xiTiAnswer[0]):
                        os.makedirs(deep + '/numImg/' + xiTiAnswer[0])
                    pictureN = len(os.listdir(deep + '/numImg/' + xiTiAnswer[0])) + 1
                    cv2.imwrite(deep + '/numImg/' + xiTiAnswer[0] + "/" + str(pictureN) + ".jpg", kouSuanCropImg)
                elif (len(re.findall(r"[0-9]", xiTiAnswer[0])) > 1) and \
                        (oralArithmeticCoor[4][j] - oralArithmeticCoor[2][j] > 2) and \
                        (oralArithmeticCoor[3][j] - oralArithmeticCoor[1][j] > 2):
                    kouSuanCropImg = ansImg[oralArithmeticCoor[2][j]:oralArithmeticCoor[4][j], \
                                     oralArithmeticCoor[1][j]:oralArithmeticCoor[3][j]]
                    if not os.path.exists(deep + '/numStringImg/' + xiTiAnswer[0]):
                        os.makedirs(deep + '/numStringImg/' + xiTiAnswer[0])
                    pictureN = len(os.listdir(deep + '/numStringImg/' + xiTiAnswer[0])) + 1
                    cv2.imwrite(deep + '/numStringImg/' + xiTiAnswer[0] + "/" + str(pictureN) + ".jpg", kouSuanCropImg)
        elif (middleNum in completionPosition):
            if (len(completionPosition[middleNum]) > 0):
                # Number of the xiaoti answer
                answerPositionNum = len(xiTiAnswer)
                if (len(xiTiAnswer) > len(completionPosition[middleNum])):
                    answerPositionNum = len(completionPosition[middleNum])
                for k in range(0, answerPositionNum):
                    if (completionPosition[middleNum][k][3] - completionPosition[middleNum][k][1] > 2) and \
                            (completionPosition[middleNum][k][2] - completionPosition[middleNum][k][0] > 2):
                        if (len(re.findall(r"[0-9]", xiTiAnswer[k])) == 1) and (len(xiTiAnswer[k]) == 1):
                            tianKongCropImg = ansImg[
                                              completionPosition[middleNum][k][1]:completionPosition[middleNum][k][3], \
                                              completionPosition[middleNum][k][0]:completionPosition[middleNum][k][2]]
                            if not os.path.exists(deep + '/numImg/' + xiTiAnswer[k]):
                                os.makedirs(deep + '/numImg/' + xiTiAnswer[k])
                            pictureN = len(os.listdir(deep + '/numImg/' + xiTiAnswer[k])) + 1
                            cv2.imwrite(deep + '/numImg/' + xiTiAnswer[k] + "/" + str(pictureN) + ".jpg",
                                        tianKongCropImg)
                        elif (len(re.findall(r"[0-9]", xiTiAnswer[k])) > 1) and (
                                len(re.findall(r"[0-9]", xiTiAnswer[k])) == len(xiTiAnswer[k])):
                            tianKongCropImg = ansImg[
                                              completionPosition[middleNum][k][1]:completionPosition[middleNum][k][3], \
                                              completionPosition[middleNum][k][0]:completionPosition[middleNum][k][2]]
                            if not os.path.exists(deep + '/numStringImg/' + xiTiAnswer[k]):
                                os.makedirs(deep + '/numStringImg/' + xiTiAnswer[k])
                            pictureN = len(os.listdir(deep + '/numStringImg/' + xiTiAnswer[k])) + 1
                            cv2.imwrite(deep + '/numStringImg/' + xiTiAnswer[k] + "/" + str(pictureN) + ".jpg",
                                        tianKongCropImg)
                        elif (len(re.findall(r"[\u4e00-\u9fa5]", xiTiAnswer[k])) == 1) and (len(xiTiAnswer[k]) == 1):
                            tianKongCropImg = ansImg[
                                              completionPosition[middleNum][k][1]:completionPosition[middleNum][k][3], \
                                              completionPosition[middleNum][k][0]:completionPosition[middleNum][k][2]]
                            if not os.path.exists(deep + '/chiImg/' + xiTiAnswer[k]):
                                os.makedirs(deep + '/chiImg/' + xiTiAnswer[k])
                            pictureN = len(os.listdir(deep + '/chiImg/' + xiTiAnswer[k])) + 1
                            cv2.imwrite(deep + '/chiImg/' + xiTiAnswer[k] + "/" + str(pictureN) + ".jpg",
                                        tianKongCropImg)
                        elif (len(re.findall(r"[\u4e00-\u9fa5]", xiTiAnswer[k])) > 1) and \
                                (len(re.findall(r"[\u4e00-\u9fa5]", xiTiAnswer[k])) == len(xiTiAnswer[k])):
                            tianKongCropImg = ansImg[
                                              completionPosition[middleNum][k][1]:completionPosition[middleNum][k][3], \
                                              completionPosition[middleNum][k][0]:completionPosition[middleNum][k][2]]
                            if not os.path.exists(deep + '/chiStringImg/' + xiTiAnswer[k]):
                                os.makedirs(deep + '/chiStringImg/' + xiTiAnswer[k])
                            pictureN = len(os.listdir(deep + '/chiStringImg/' + xiTiAnswer[k])) + 1
                            cv2.imwrite(deep + '/chiStringImg/' + xiTiAnswer[k] + "/" + str(pictureN) + ".jpg",
                                        tianKongCropImg)
                        elif (len(re.findall(r"[\u0041-\u005a,\u0061-\u007a]", xiTiAnswer[k])) == 1) and (
                                len(xiTiAnswer[k]) == 1):
                            tianKongCropImg = ansImg[
                                              completionPosition[middleNum][k][1]:completionPosition[middleNum][k][3], \
                                              completionPosition[middleNum][k][0]:completionPosition[middleNum][k][2]]
                            if not os.path.exists(deep + '/engImg/' + xiTiAnswer[k]):
                                os.makedirs(deep + '/engImg/' + xiTiAnswer[k])
                            pictureN = len(os.listdir(deep + '/engImg/' + xiTiAnswer[k])) + 1
                            cv2.imwrite(deep + '/engImg/' + xiTiAnswer[k] + "/" + str(pictureN) + ".jpg",
                                        tianKongCropImg)
                        elif (len(re.findall(r"[\u0041-\u005a,\u0061-\u007a]", xiTiAnswer[k])) > 1) and \
                                (len(re.findall(r"[\u0041-\u005a,\u0061-\u007a]", xiTiAnswer[k])) == len(
                                    xiTiAnswer[k])):
                            tianKongCropImg = ansImg[
                                              completionPosition[middleNum][k][1]:completionPosition[middleNum][k][3], \
                                              completionPosition[middleNum][k][0]:completionPosition[middleNum][k][2]]
                            if not os.path.exists(deep + '/engStringImg/' + xiTiAnswer[k]):
                                os.makedirs(deep + '/engStringImg/' + xiTiAnswer[k])
                            pictureN = len(os.listdir(deep + '/engStringImg/' + xiTiAnswer[k])) + 1
                            cv2.imwrite(deep + '/engStringImg/' + xiTiAnswer[k] + "/" + str(pictureN) + ".jpg",
                                        tianKongCropImg)
                        elif (len(xiTiAnswer[k]) > 3):
                            tianKongCropImg = ansImg[
                                              completionPosition[middleNum][k][1]:completionPosition[middleNum][k][3], \
                                              completionPosition[middleNum][k][0]:completionPosition[middleNum][k][2]]
                            # cv2.imwrite(sign_string_img_path+"/sign_"+str(daTiNum+titleMin)+"." + str(middleNum) +"."+str(k+1)+".jpg", tianKongCropImg)
                        else:
                            tianKongCropImg = ansImg[
                                              completionPosition[middleNum][k][1]:completionPosition[middleNum][k][3], \
                                              completionPosition[middleNum][k][0]:completionPosition[middleNum][k][2]]
                            if not os.path.exists(deep + '/signImg/' + xiTiAnswer[k]):
                                os.makedirs(deep + '/signImg/' + xiTiAnswer[k])
                            pictureN = len(os.listdir(deep + '/signImg/' + xiTiAnswer[k])) + 1
                            cv2.imwrite(deep + '/signImg/' + xiTiAnswer[k] + "/" + str(pictureN) + ".jpg",
                                        tianKongCropImg)

        elif (middleNum in choicePosition):
            if (len(choicePosition[middleNum]) > 0):
                # Number of the xiaoti answer
                answerPositionNum = len(xiTiAnswer)
                if (len(xiTiAnswer) > len(choicePosition[middleNum])):
                    answerPositionNum = len(choicePosition[middleNum])
                for k in range(0, answerPositionNum):
                    if (choicePosition[middleNum][k][3] - choicePosition[middleNum][k][1] > 2) and \
                            (choicePosition[middleNum][k][2] - choicePosition[middleNum][k][0] > 2):
                        if (len(re.findall(r"[\u0041-\u005a,\u0061-\u007a]", xiTiAnswer[k])) == 1) and (
                                len(xiTiAnswer[k]) == 1):
                            choiceCropImg = ansImg[choicePosition[middleNum][k][1]:choicePosition[middleNum][k][3], \
                                            choicePosition[middleNum][k][0]:choicePosition[middleNum][k][2]]
                            if not os.path.exists(deep + '/engImg/' + xiTiAnswer[k]):
                                os.makedirs(deep + '/engImg/' + xiTiAnswer[k])
                            pictureN = len(os.listdir(deep + '/engImg/' + xiTiAnswer[k])) + 1
                            cv2.imwrite(deep + '/engImg/' + xiTiAnswer[k] + "/" + str(pictureN) + ".jpg", choiceCropImg)
                        elif (len(re.findall(r"[\u0041-\u005a,\u0061-\u007a]", xiTiAnswer[k])) > 1) and \
                                (len(re.findall(r"[\u0041-\u005a,\u0061-\u007a]", xiTiAnswer[k])) == len(
                                    xiTiAnswer[k])):
                            choiceCropImg = ansImg[choicePosition[middleNum][k][1]:choicePosition[middleNum][k][3], \
                                            choicePosition[middleNum][k][0]:choicePosition[middleNum][k][2]]
                            if not os.path.exists(deep + '/engStringImg/' + xiTiAnswer[k]):
                                os.makedirs(deep + '/engStringImg/' + xiTiAnswer[k])
                            pictureN = len(os.listdir(deep + '/engStringImg/' + xiTiAnswer[k])) + 1
                            cv2.imwrite(deep + '/engStringImg/' + xiTiAnswer[k] + "/" + str(pictureN) + ".jpg",
                                        choiceCropImg)
                        elif (len(xiTiAnswer[k]) > 3):
                            choiceCropImg = ansImg[choicePosition[middleNum][k][1]:choicePosition[middleNum][k][3], \
                                            choicePosition[middleNum][k][0]:choicePosition[middleNum][k][2]]
                            # cv2.imwrite(sign_string_img_path+"/sign_"+str(daTiNum+titleMin)+"." + str(middleNum) +"."+str(k+1)+".jpg", choiceCropImg)
                        else:
                            choiceCropImg = ansImg[choicePosition[middleNum][k][1]:choicePosition[middleNum][k][3], \
                                            choicePosition[middleNum][k][0]:choicePosition[middleNum][k][2]]
                            if not os.path.exists(deep + '/signImg/' + xiTiAnswer[k]):
                                os.makedirs(deep + '/signImg/' + xiTiAnswer[k])
                            pictureN = len(os.listdir(deep + '/signImg/' + xiTiAnswer[k])) + 1
                            cv2.imwrite(deep + '/signImg/' + xiTiAnswer[k] + "/" + str(pictureN) + ".jpg",
                                        choiceCropImg)
        elif (middleNum in judgementPosition):
            if (len(judgementPosition[middleNum]) > 0):
                # Number of the xiaoti answer
                answerPositionNum = len(xiTiAnswer)
                if (len(xiTiAnswer) > len(judgementPosition[middleNum])):
                    answerPositionNum = len(judgementPosition[middleNum])
                for k in range(0, answerPositionNum):
                    if (judgementPosition[middleNum][k][3] - judgementPosition[middleNum][k][1] > 2) and \
                            (judgementPosition[middleNum][k][2] - judgementPosition[middleNum][k][0] > 2):
                        judgementCropImg = ansImg[judgementPosition[middleNum][k][1]:judgementPosition[middleNum][k][3], \
                                           judgementPosition[middleNum][k][0]:judgementPosition[middleNum][k][2]]
                        if not os.path.exists(deep + '/signImg/' + xiTiAnswer[k]):
                            os.makedirs(deep + '/signImg/' + xiTiAnswer[k])
                        pictureN = len(os.listdir(deep + '/signImg/' + xiTiAnswer[k])) + 1
                        cv2.imwrite(deep + '/signImg/' + xiTiAnswer[k] + "/" + str(pictureN) + ".jpg", judgementCropImg)


if __name__ == '__main__':
    outJsonPath = '/home/easyxue/demos/oral/json/'
    print("AI Analyzer Module...")
    completeJsonData = []
    inputJsonData = {}
    coordinateOfTitlesAll = []
    coorOfPartitionsAll = []
    xiaoTiAll = []
    deep = ""

    try:
        options, args = getopt.getopt(sys.argv[1:], 'sj:o:d:', ['script', 'json=', 'outjson=', 'deep='])
    except getopt.GetoptError:
        print("no valid input argument, exit...")
        sys.exit()

    for option, value in options:
        if option in ('-j', '--j'):
            Data = value
            # print('inputJsonData is :{}'.format(Data))
            # print ('ytpe....',type(Data))
            inputJsonData = json.loads(Data)
        if option in ('-o', '--o'):
            outJsonPath = value
            # print('outJsonPath is :{}'.format(outJsonPath))
        if option in ('-d', '--d'):
            deep = value
            # print('deep is :{}'.format(deep))
            if not os.path.exists(deep):
                os.makedirs(deep)
            if not os.path.exists(deep + '/numImg'):
                os.makedirs(deep + '/numImg')
            if not os.path.exists(deep + '/numStringImg'):
                os.makedirs(deep + '/numStringImg')
            if not os.path.exists(deep + '/engImg'):
                os.makedirs(deep + '/engImg')
            if not os.path.exists(deep + '/engStringImg'):
                os.makedirs(deep + '/engStringImg')
            if not os.path.exists(deep + '/chiImg'):
                os.makedirs(deep + '/chiImg')
            if not os.path.exists(deep + '/chiStringImg'):
                os.makedirs(deep + '/chiStringImg')
            if not os.path.exists(deep + '/signImg'):
                os.makedirs(deep + '/signImg')

    # One more model paper will be passed in
    for modlePaperNum in range(0, len(inputJsonData['modelPaper'])):
        # Download modelpaper
        urllib.request.urlretrieve(inputJsonData['modelPaper'][modlePaperNum], \
                                   modelPaperPath + inputJsonData['modelPaper'][modlePaperNum].split('/')[-1])
        # Get the titles position of the dati
        paper = sort_position_0603.Paper(modelPaperPath + inputJsonData['modelPaper'][modlePaperNum].split('/')[-1])
        paper.get_dati_partitions()
        coordinateOfTitlesAll.append(paper.coorOfDaTiTitles)
        coorOfPartitionsAll.append(paper.coorOfDaTiPartitions)
    
        # 获取当前试卷面的大题题号的最小值
        numSpiltMin = min(list(paper.coorOfDaTiTitles.keys())) 
        """
        如果当前试卷左上角存在大题跨面，则大题题号最小值减1。
        （比如当前面最小大题题号是第五大题，但是当前面左上角有一部分第四大题的内容，且无第四大题题号，此时numSpiltMin应减1）
        原因是大小题分割模块，会把除第一面外的当前面，其大题题号最小值以上的部分都当做是上一个大题的区域.
        这一左上角区域可能是空白区域，或者是上一面最后一大题的剩余部分，但无论是哪一种情况，numSpiltMin都要减1              
        """
        if (numSpiltMin != 1):
            numSpiltMin = numSpiltMin - 1
        numSpiltMax = max(list(paper.coorOfDaTiTitles.keys()))
        if (len(inputJsonData['answerList']) != 0):
            paperAnswerL = {} #存放当前试卷面存在的大题其答案库数据
            # 如果母卷的内容总共只有一面
            if (len(inputJsonData['modelPaper']) == 1):
                paperAnswerL['answerList'] = inputJsonData['answerList'][0:]
            else:
                paperAnswerL['answerList'] = inputJsonData['answerList'][numSpiltMin - 1:numSpiltMax]
            paper.get_xiaoti_partitions(paperAnswerL)
            xiaoTiAll.append(paper.coorOfXiaoTiPartitions)
    
    #key为int型
    # coordinateOfTitlesAll = [{1: [228, 897, 893, 949], 2: [232, 1646, 729, 1699], 3: [2388, 1000, 2799, 1053], 4: [2392, 1444, 3530, 1498]},\
    # {5: [244, 475, 768, 529], 6: [237, 2034, 899, 2089], 7: [2390, 1943, 2844, 1996]}]

    #key为string型 当前面的最后一个大题的key都是"题号-1"的形式，反面如果存在该大题区域，则有"题号-2"形式的key，没有则无
    # coorOfPartitionsAll = [{"1": [0, 949, 2182, 1646], "2-1": [0, 1699, 2182, 2935], "2-2": [2182, 0, 4364, 1000], "3": [2182, 1053, 4364, 1444], "4-1": [2182, 1498, 4364, 2935]},\
    # {"4-2": [0, 0, 2187, 475], "5": [0, 529, 2187, 2034], "6-1": [0, 2089, 2187, 2936], "6-2": [2187, 0, 4375, 1943], "7": [2187, 1996, 4375, 2936]}]

    # xiaoTiAll = [{4: {1: [2378, 1531, 4364, 2022], 2: [2378, 2022, 4364, 2935]}, 3: {1: [2378, 1053, 4364, 1444]}, 2: {1: [218, 1716, 2182, 2092], 2: [218, 2092, 2182, 2184], 3: [218, 2184, 2182, 2375], 4: [218, 2375, 2182, 2656], 5: [218, 2656, 2182, 2935], 6: [2378, 210, 4364, 535], 7: [2378, 535, 4364, 893], 8: [2378, 893, 4364, 1000]}, 1: {1: [218, 949, 2182, 1646]}},\
    # {7: {1: [2380, 2009, 4375, 2465], 2: [2380, 2465, 4375, 2936]}, 6: {1: [234, 2101, 2187, 2936], 2: [2380, 206, 4375, 606], 3: [2380, 606, 4375, 1001], 4: [2380, 1001, 4375, 1484], 5: [2380, 1484, 4375, 1943]}, 5: {1: [234, 544, 2187, 977], 2: [234, 977, 2187, 1497], 3: [234, 1497, 2187, 2034]}, 4: {3: [234, 196, 2187, 475]}}]

    xueHao = xuehao.xuehao(modelPaperPath + inputJsonData['modelPaper'][0].split('/')[-1])
    print('xuehao:\n{}'.format(xueHao))
    print('coordinateOfTitlesAll: \n{}'.format(coordinateOfTitlesAll))
    print('coorOfPartitionsAll:\n{}'.format(coorOfPartitionsAll))
    print('xiaoTiAll:\n{}'.format(xiaoTiAll))

    for studentNum in range(0, len(inputJsonData['paperList'])):
        studentTempUuid = uuid.uuid1()
        for paperIdx in range(0, len(inputJsonData['paperList'][studentNum])):
            jsonData = copy.deepcopy(inputJsonData)  # create a deep copy of it
            print('processing paper {}'.format(paperIdx))
            print('processing paper: {}'.format(jsonData['paperList'][studentNum][paperIdx]))
            # Papers name
            modelPaperName = jsonData['modelPaper'][paperIdx].split('/')[-1]
            paperListName = jsonData['paperList'][studentNum][paperIdx].split('/')[-1]
            # Papers path
            ansImgPath = paperListPath + paperListName
            realImgPath = modelPaperPath + modelPaperName
            # Download the answer papers
            urllib.request.urlretrieve(jsonData['paperList'][studentNum][paperIdx], ansImgPath)

            try:
                aImg = Image.open(ansImgPath)
            except IOError:
                print('cannot open', ansImgPath)
            else:
                oralArithmeticCoorAll = []
                completionAll = []
                choiceAll = []
                judgementAll = []

                # Create outJson
                if (len(jsonData['answerList']) != 0):
                    outJson = createOutJson(jsonData, xiaoTiAll[paperIdx], coorOfPartitionsAll[paperIdx], paperIdx)
                else:
                    outJson = createOutJsonNoAnswerList(jsonData, coorOfPartitionsAll[paperIdx],
                                                        coordinateOfTitlesAll[paperIdx])
                # Picture preprocessing
                cvImg = preprocess.separateAns(realImgPath, ansImgPath)
                cv2.imwrite(paperListPath + 'separate_' + str(studentNum) + '_' + paperListName, cvImg)
                ansImg = cv2.imread(paperListPath + 'separate_' + str(studentNum) + '_' + paperListName)
                # Picture alignment
                cvImg1 = preprocess.align(realImgPath, ansImgPath)
                cv2.imwrite(paperListPath + str(studentNum) + '_' + paperListName, cvImg1)
                ansImg1 = cv2.imread(paperListPath + str(studentNum) + '_' + paperListName)
                # Answer pictures
                ansImg2 = cv2.imread(ansImgPath)
                modelPaperImg = cv2.imread(realImgPath)

                titleMin = outJson['answerList'][0]['sort']

                if (len(inputJsonData['answerList']) != 0):
                    timeN = time.strftime('%d-%H-%M-%S', time.localtime())
                    timeN = timeN + "_" + paperListName + "_"
                    num_img_path = imgStorePath + timeN + 'num_img'
                    sign_img_path = imgStorePath + timeN + 'sign_img'
                    eng_img_path = imgStorePath + timeN + 'eng_img'
                    chi_img_path = imgStorePath + timeN + 'chi_img'

                    num_string_img_path = imgStorePath + timeN + 'num_string_img'
                    sign_string_img_path = imgStorePath + timeN + 'sign_string_img'
                    eng_string_img_path = imgStorePath + timeN + 'eng_string_img'
                    chi_string_img_path = imgStorePath + timeN + 'chi_string_img'

                    os.mkdir(num_img_path)
                    os.mkdir(sign_img_path)
                    os.mkdir(eng_img_path)
                    os.mkdir(chi_img_path)
                    os.mkdir(num_string_img_path)
                    os.mkdir(sign_string_img_path)
                    os.mkdir(eng_string_img_path)
                    os.mkdir(chi_string_img_path)

                    for daTiNum in range(0, len(outJson['answerList'])):
                        oralArithmeticCoor = []
                        completionPosition = {}
                        choicePosition = {}
                        judgementPosition = {}                        
                        completionExist = 0 # 判断填空题是否存在
                        # Get the regional coordinates of the answers to the completion
                        for i in range(0, len(outJson['answerList'][daTiNum]['paperQuestionList'])):
                            if (outJson['answerList'][daTiNum]['paperQuestionList'][i]["type"] == 20) and \
                                    (outJson['answerList'][daTiNum]['paperQuestionList'][i]["autoMarkFlag"] == 1):
                                completionExist = 1;
                        # Get the coordinates of the answer area of the oral arithmetic
                        if (outJson['answerList'][daTiNum]['paperQuestionList'][0]["type"] == 11) and \
                                (outJson['answerList'][daTiNum]['paperQuestionList'][0]["autoMarkFlag"] == 1) and \
                                (len(xiaoTiAll[paperIdx][daTiNum + titleMin]) == 1):
                            # 如果key是以"题号-1"的形式存在
                            if (str(daTiNum + titleMin) + '-1' in coorOfPartitionsAll[paperIdx]):
                                oralArithmeticArea = coorOfPartitionsAll[paperIdx][str(daTiNum + titleMin) + '-1']
                            else:
                                oralArithmeticArea = coorOfPartitionsAll[paperIdx][str(daTiNum + titleMin)]
                            oralArithmeticBoxPosition = kousuanti.single_image(realImgPath, oralArithmeticArea)
                            calAnsPosT = oral1_integrate1.calAnsPos(oralArithmeticArea, ansImg, modelPaperImg)
                            oralArithmeticCoor = calAnsPosT.answer_zone(oralArithmeticBoxPosition, oralArithmeticArea)
                            print('KOUSUAN oralArithmeticBoxPosition', oralArithmeticBoxPosition)
                        # Get the coordinates of the answer area of the oral completion
                        elif (completionExist) and (daTiNum + titleMin in xiaoTiAll[paperIdx]):
                            completionZhongTi = xiaoTiAll[paperIdx][daTiNum + titleMin]
                            gapFillAnsPos = gapFillAnsPos3.gapFillAnsPos(cvImg, cvImg1, modelPaperImg, outJson)
                            completionPosition = gapFillAnsPos.ansSelect(realImgPath, completionZhongTi, daTiNum)
                        # Get the coordinates of the answer area of the oral choice
                        elif (outJson['answerList'][daTiNum]['paperQuestionList'][0]["type"] == 30) and \
                                (outJson['answerList'][daTiNum]['paperQuestionList'][0]["autoMarkFlag"] == 1):
                            choiceZhongTi = xiaoTiAll[paperIdx][daTiNum + titleMin]
                            choiceAnsPos = gapFillAnsPos3.gapFillAnsPos(cvImg, cvImg1, modelPaperImg, outJson)
                            choicePosition = choiceAnsPos.ansSelect(realImgPath, choiceZhongTi, daTiNum)
                            # Get the coordinates of the answer area of the oral judgement
                        elif (outJson['answerList'][daTiNum]['paperQuestionList'][0]["type"] == 40) and \
                                (outJson['answerList'][daTiNum]['paperQuestionList'][0]["autoMarkFlag"] == 1):
                            judgementZhongTi = xiaoTiAll[paperIdx][daTiNum + titleMin]
                            judgementAnsPos = gapFillAnsPos3.gapFillAnsPos(cvImg, cvImg1, modelPaperImg, outJson)
                            judgementPosition = judgementAnsPos.ansSelect(realImgPath, judgementZhongTi, daTiNum)

                        oralArithmeticCoorAll.append(oralArithmeticCoor)
                        completionAll.append(completionPosition)
                        choiceAll.append(choicePosition)
                        judgementAll.append(judgementPosition)
                        # Screenshot
                        screenshot(outJson, ansImg, daTiNum, titleMin, oralArithmeticCoor, completionPosition,
                                   choicePosition, judgementPosition)
                        # Deep learning screenshots
                        if len(deep) > 0:
                            screenshotDeepL(deep, outJson, ansImg, daTiNum, titleMin, oralArithmeticCoor,
                                            completionPosition, choicePosition, judgementPosition)
                    print('oralArithmeticCoorAll......', oralArithmeticCoorAll)
                    print('completionAll........ ', completionAll)
                    print('choiceAll..........', choiceAll)
                    print('judgementAll.........', judgementAll)
                    # Call AI Analyzer Module
                    os.system('python3 sigle_char_recognize2.py ' + timeN)
                    # os.system('python3 string_char_recognize2.py ' + timeN)
                    
                    # Obtaining the recognition result of students' answers
                    numResult, numStringResult, signResult, engResult, chiResult = analyzerResult();
                    
                    # print('numResult = {}'.format(numResult))
                    # print('numStringResult = {}'.format(numStringResult))
                    # print('signResult = {}'.format(signResult))
                    # print('engResult = {}'.format(engResult))
                    # print('chiResult = {}'.format(chiResult))

                # Fill the corresponding data in the frame of the outJson
                outJson['resultFile']['name'] = paperListPath + str(studentNum) + '_' + paperListName
                # Get image size
                inputImg = Image.open(outJson['resultFile']['name'])
                inputImgWidth, inputImgHeight = inputImg.size
                outJson['resultFile']['size'] = [inputImgWidth, inputImgHeight]
                outJson['resultFile']['origPaperUrl'] = outJson['paperList'][studentNum][paperIdx]
                
                # Get xuehao position
                if (len(xueHao) > 0):
                    outJson['studentInfo']['position']['LX'] = xueHao[0]
                    outJson['studentInfo']['position']['LY'] = xueHao[1]
                    outJson['studentInfo']['position']['RX'] = xueHao[2]
                    outJson['studentInfo']['position']['RY'] = xueHao[3]
                outJson["studentInfo"]["studentTempUuid"] = str(studentTempUuid)
                outJson["studentInfo"]["answerPaperIndex"] = paperIdx
                
                # Fill in the coordinates of the dati
                daTiCoordinateW(outJson, titleMin, paperIdx, coorOfPartitionsAll[paperIdx],
                                coordinateOfTitlesAll[paperIdx], completeJsonData)
                
                # The number of answers is not based on the length of questionID in input json data.
                # We return the number of answers actually detected by the module instead.
                if (len(inputJsonData['answerList']) != 0):
                    # Write the coordinates of each subject
                    zhongTiPositionW(outJson, xiaoTiAll[paperIdx])
                    for daTiNum in range(0, len(outJson['answerList'])):
                        if (len(oralArithmeticCoorAll[daTiNum]) > 0):
                            # Fill in the data of oral arithmetic
                            oralArithmeticAnswerL(outJson, daTiNum, titleMin, oralArithmeticCoorAll[daTiNum], numResult,
                                                  numStringResult)
                        elif (len(completionAll[daTiNum]) > 0):
                            # Fill in the data of completion
                            completionAnswerL(outJson, daTiNum, titleMin, completionAll[daTiNum], \
                                              numResult, numStringResult, signResult, engResult, chiResult)
                        elif (len(choiceAll[daTiNum]) > 0):
                            # Fill in the data of choice
                            choiceAnswerL(outJson, daTiNum, titleMin, choiceAll[daTiNum], signResult, engResult)
                        elif (len(judgementAll[daTiNum]) > 0):
                            # Fill in the data of judgement
                            judgementAnswerL(outJson, daTiNum, titleMin, judgementAll[daTiNum], signResult)

                del outJson['modelPaper']
                del outJson['paperList']
                completeJsonData.append(outJson)

    currentTime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    # os.mknod(outJsonPath + currentTime + '_' + modelPaperName + '_' + str(paperIdx) + '_out.json')

    with open(outJsonPath + currentTime + '_' + modelPaperName + '_' + str(paperIdx) + '_out.json', "w",
              encoding="utf-8") as inData:
        inData.write(json.dumps(completeJsonData, sort_keys=False, indent=4))

    # resultPath = outJsonPath+currentTime+"_out.json"
    # url = 'http://127.0.0.1:8090/analysis/aiServer/finish?resultPath=%s'%resultPath
    # response = urllib.request.urlopen(url)
    # print ((response.read()).decode('utf-8'))
