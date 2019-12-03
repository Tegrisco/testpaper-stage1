import sys
import json
import cv2
import os
import urllib.request

def extractJsonFromFile(jsonFilePath):
    '''
    # count the number of lines in input json file
    with open(jsonFilePath) as f:
        numOfLines = sum(1 for _ in f) 
    assert numOfLines == 1, 'Valid json file should contain only one entry! It has {}.'.format(numOfLines)

    jsonFile = open(jsonFilePath, 'r')
    return json.loads(jsonFile.readlines())
    '''

    jsonString = ''
    with open(jsonFilePath) as f:
        for line in f:
            jsonString = jsonString + line

    # remove \r, \n, \t and whitespaces
    jsonString = jsonString.replace('\n', '')
    jsonString = jsonString.replace('\t', '')
    jsonString = jsonString.replace('\r', '')
    jsonString = jsonString.replace(' ', '')
    
    return jsonString

'''
    Function description: 
      Compute accuracy number for all answers with single digit number.
    
    Parameters: json Data 
    Return value: accuracy value
'''
#def getAccuracyNum(jsonData):


'''
    Function description: 
      Put a boundary box around answers according to position coordinates
    in json data. 
      Use different colors to indicate correctness.
      Output file will be saved as STUDENT_TEMP_UUID_PAPER_INDEX.jpg.
    
    Parameters: 
      json Data 
      output directory
    Return value: none
'''
def markAllAnswerBoxes(jsonData, outputDir):

    def allPositionEqualZero(positionList):
        assert type(positionList) == list, 'Function input parameter invalid!'
        assert type(positionList[0]) == dict, 'Function input parameter invalid!'
        if positionList[0]['LX'] == 0 and positionList[0]['LY'] == 0 and \
           positionList[0]['RX'] == 0 and positionList[0]['RY'] == 0:
            return True
        else:
            return False

    for paperInfo in jsonData:
        #print (paperInfo)

        #print ('student uuid: {}'.format(paperInfo['studentInfo']['studentTempUuid']))
        #print ('paper index: {}'.format(paperInfo['studentInfo']['answerPaperIndex']))
        outputFileName = paperInfo['studentInfo']['studentTempUuid'] + '-' + str(paperInfo['studentInfo']['answerPaperIndex'])
        print ('outputFileName: {}'.format(outputFileName))

        # use answer paper url instead of hard-coding file name
        # open the original answer paper for editing
        #origPaperUrl = paperInfo['resultFile']['origPaperUrl']
        #print ('Ready to download {}'.format(origPaperUrl))
        #paperFileName = origPaperUrl.split('/')[-1]
        #urllib.request.urlretrieve(origPaperUrl, paperFileName) 
        #mark_img = cv2.imread(paperFileName) 

        # use answer paper url instead of hard-coding file name
        # open the final adjusted answer paper for editing
        paperFileName = paperInfo['resultFile']['name']
        mark_img = cv2.imread(paperFileName) 

        for daTiInfo in paperInfo['answerList']:
            #print ('Dati info: {}'.format(daTiInfo))
            for zhongTiInfo in daTiInfo['paperQuestionList']:
                #print ('Zhongti info: {}'.format(zhongTiInfo))
                for xiaoTiResultInfo in zhongTiInfo['result']:
                    print ('Xiaoti result = {}'.format(xiaoTiResultInfo['recognitionResult']))
                    positionList = xiaoTiResultInfo['position']
                    if xiaoTiResultInfo['resultFlag'] == True:
                        cv2.rectangle(mark_img, (positionList[0]['LX'], positionList[0]['LY']), \
                                      (positionList[0]['RX'], positionList[0]['RY']),\
                                      (0, 255, 0), 4\
                                     )
                        #print ('Found one correct answer.')
                    elif xiaoTiResultInfo['resultFlag'] == False and \
                         allPositionEqualZero(positionList) == False:
                        cv2.rectangle(mark_img, (positionList[0]['LX'], positionList[0]['LY']), \
                                      (positionList[0]['RX'], positionList[0]['RY']),\
                                      (0, 0, 255), 4\
                                     )
                        cv2.putText(mark_img, xiaoTiResultInfo['recognitionResult'], \
                                    (positionList[0]['RX'], positionList[0]['RY']),\
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4\
                                   )
                        #print('Found one wrong answer.')
                    else:
                        print('answer area not found.')

        # draw and save image
        if not os.path.exists(outputDir+'_output'):
            print ('output to directory {}'.format(outputDir+'_output'))
            os.makedirs(outputDir+'_output')
        cv2.imwrite('./'+ outputDir + '_output/' + outputFileName+'.jpg', mark_img)
    
     
'''
    Function description: 
      Put a boundary box around all positions according to position coordinates
    in json data. 
      Use different colors to indicate correctness.
      Output file will be saved as STUDENT_TEMP_UUID_PAPER_INDEX.jpg.
    
    Parameters: 
      json Data 
      output directory
    Return value: none
'''
def markAllBoxes(jsonData, outputDir):

    def allPositionEqualZero(positionList):
        assert type(positionList) == list, 'Function input parameter invalid!'
        assert type(positionList[0]) == dict, 'Function input parameter invalid!'
        if positionList[0]['LX'] == 0 and positionList[0]['LY'] == 0 and \
           positionList[0]['RX'] == 0 and positionList[0]['RY'] == 0:
            return True
        else:
            return False

    for paperInfo in jsonData:
        #print (paperInfo)

        #print ('student uuid: {}'.format(paperInfo['studentInfo']['studentTempUuid']))
        #print ('paper index: {}'.format(paperInfo['studentInfo']['answerPaperIndex']))
        outputFileName = paperInfo['studentInfo']['studentTempUuid'] + '-' + str(paperInfo['studentInfo']['answerPaperIndex'])
        print ('outputFileName: {}'.format(outputFileName))

        # use answer paper url instead of hard-coding file name
        # open the original answer paper for editing
        #origPaperUrl = paperInfo['resultFile']['origPaperUrl']
        #print ('Ready to download {}'.format(origPaperUrl))
        #paperFileName = origPaperUrl.split('/')[-1]
        #urllib.request.urlretrieve(origPaperUrl, paperFileName) 
        #mark_img = cv2.imread(paperFileName) 

        # use answer paper url instead of hard-coding file name
        # open the final adjusted answer paper for editing
        paperFileName = paperInfo['resultFile']['name']
        mark_img = cv2.imread(paperFileName) 

        for daTiInfo in paperInfo['answerList']:
            #print ('Dati info: {}'.format(daTiInfo))
 
            # put boxes on Dati boundary
            #for daTiPosition in daTiInfo['position']:
            #    cv2.rectangle(mark_img, (daTiPosition['LX'], daTiPosition['LY']), \
            #              (daTiPosition['RX'], daTiPosition['RY']),\
            #              (127, 127, 127), 4\
            #             )

            for zhongTiInfo in daTiInfo['paperQuestionList']:
                #print ('Zhongti info: {}'.format(zhongTiInfo))
           
                 
                # put boxes on Zhongti boundary
                for zhongTiPosition in zhongTiInfo['position']:
                    #print ('zhongTiPosition: {}'.format(zhongTiPosition))
                    cv2.rectangle(mark_img, (zhongTiPosition['LX'], zhongTiPosition['LY']), \
                              (zhongTiPosition['RX'], zhongTiPosition['RY']),\
                              (255, 0, 0), 4\
                             )

                # do nothing for zhongti with autoMarkFlag equal to zero
                #if zhongTiInfo['autoMarkFlag'] == 0:
                #    continue
                

                for xiaoTiResultInfo in zhongTiInfo['result']:
                    #print ('Xiaoti result = {}'.format(xiaoTiResultInfo['recognitionResult']))
                    positionList = xiaoTiResultInfo['position']
                    if xiaoTiResultInfo['resultFlag'] == True:
                        cv2.rectangle(mark_img, (positionList[0]['LX'], positionList[0]['LY']), \
                                      (positionList[0]['RX'], positionList[0]['RY']),\
                                      (0, 255, 0), 4\
                                     )
                        #print ('Found one correct answer.')
                    elif xiaoTiResultInfo['resultFlag'] == False and \
                         allPositionEqualZero(positionList) == False:
                        cv2.rectangle(mark_img, (positionList[0]['LX'], positionList[0]['LY']), \
                                      (positionList[0]['RX'], positionList[0]['RY']),\
                                      (0, 0, 255), 4\
                                     )
                        cv2.putText(mark_img, xiaoTiResultInfo['recognitionResult'], \
                                    (positionList[0]['LX'], 2*positionList[0]['RY']-positionList[0]['LY']),\
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4\
                                   )
                        #print('Found one wrong answer.')
                    else:
                        pass
                        #print('answer area not found.')

        # draw and save image
        if not os.path.exists(outputDir+'_output'):
            print ('output to directory {}'.format(outputDir+'_output'))
            os.makedirs(outputDir+'_output')
        cv2.imwrite(outputDir + '_output/' + outputFileName+'.jpg', mark_img)

'''
    Function description: 
      Calculate accuracy rate of all answers with autoMarkFlag equal to 1
    in json data. 
    
    Parameters: 
      json Data 
    Return value: none
'''
def getAccuracyOfAll(jsonData):

    def allPositionEqualZero(positionList):
        assert type(positionList) == list, 'Function input parameter invalid!'
        assert type(positionList[0]) == dict, 'Function input parameter invalid!'
        if positionList[0]['LX'] == 0 and positionList[0]['LY'] == 0 and \
           positionList[0]['RX'] == 0 and positionList[0]['RY'] == 0:
            return True
        else:
            return False

    #numOfStudent = len(jsonData['paperList'])
    #print ('Number of students in json: {}'.format(numOfStudent))
    
    numOfTotalAnswer = 0
    numOfCorrectAnswer = 0

    for paperInfo in jsonData:
        #print (paperInfo)
        #print ('paper index: {}'.format(paperInfo['studentInfo']['answerPaperIndex']))

        for daTiInfo in paperInfo['answerList']:
            #print ('Dati info: {}'.format(daTiInfo))
            for zhongTiInfo in daTiInfo['paperQuestionList']:
                # do nothing for zhongti with autoMarkFlag equal to zero
                if zhongTiInfo['autoMarkFlag'] == 0:
                    continue

                # remove top-left and top-right blank area
                positionList = zhongTiInfo['position']
                if len(positionList) == 1 and positionList[0]['LY'] == 0:
                    continue 

                for xiaoTiResultInfo in zhongTiInfo['result']:
                    #print ('Xiaoti result = {}'.format(xiaoTiResultInfo['recognitionResult']))
                    

                    if xiaoTiResultInfo['resultFlag'] == True:
                        print ('Found one correct answer.')
                        numOfTotalAnswer +=1
                        numOfCorrectAnswer +=1
                    elif xiaoTiResultInfo['resultFlag'] == False and \
                         allPositionEqualZero(positionList) == False:
                        print('Found one wrong answer.')
                        numOfTotalAnswer +=1
                    else:
                        print('answer area not found.')
                        numOfTotalAnswer +=1

        
        if paperInfo['studentInfo']['answerPaperIndex'] == 1:
            print ("Paper Idx {}: total answer= {} correct answer= {} accuracy = {:.3f}".format(\
               paperInfo['studentInfo']['answerPaperIndex'], numOfTotalAnswer, numOfCorrectAnswer,\
               numOfCorrectAnswer/numOfTotalAnswer)
              )
            numOfTotalAnswer = 0
            numOfCorrectAnswer = 0

if __name__ == '__main__':
    print('Mark results on answer papers...')
    print('Usage example: python3.6 mark_result.py XXX.json')

    if len(sys.argv) >= 2:
        jsonFileName = sys.argv[1]
        print('Json File {} to be processed...'.format(jsonFileName))
    else:
        print('no valid input argument, exit...')
        sys.exit()

    inputJsonString = extractJsonFromFile(jsonFileName)
    jsonData = json.loads(inputJsonString)
    #print ('json data is {}'.format(jsonData))

    #markAllAnswerBoxes(jsonData, jsonFileName)
    
    markAllBoxes(jsonData, jsonFileName)

    accuracyOfAll = getAccuracyOfAll(jsonData)

    #accuracyNumString = getAccuracyNumString(jsonData)
