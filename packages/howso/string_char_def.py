# uncompyle6 version 3.5.1
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.7.4 (default, Sep  7 2019, 18:27:02) 
# [Clang 10.0.1 (clang-1001.0.46.4)]
# Embedded file name: string_char_def.py
# Compiled at: 2019-09-12 11:31:39
# Size of source mod 2**32: 5070 bytes
"""
Created on Thu May  9 23:17:44 2019

@author: howso
"""
import os, time, torch, pandas as pd
from PIL import Image
import howso.net as Net, howso.convert as cvt, howso.dataset as dt, howso.Config as conf
from torch.autograd import Variable
import howso.num_alphabets as numalpha, howso.sign_alphabets as signalpha, howso.eng_alphabets as engalpha, howso.chi_alphabets as chialpha, howso.point_alphabets as pointalpha, howso.douhao_alphabets as douhaoalpha
num_alphabet = numalpha.alphabet
num_class = len(num_alphabet) + 1
sign_alphabet = signalpha.alphabet
sign_class = len(sign_alphabet) + 1
eng_alphabet = engalpha.alphabet
eng_class = len(eng_alphabet) + 1
chi_alphabet = chialpha.alphabet
chi_class = len(chi_alphabet) + 1
point_alphabet = pointalpha.alphabet
point_class = len(point_alphabet) + 1
douhao_alphabet = douhaoalpha.alphabet
douhao_class = len(douhao_alphabet) + 1

def init(num_model_path_val, sign_model_path_val, eng_model_path_val, chi_model_path_val):
    global chi_string_model_path
    # global douhao_string_model_path
    global eng_string_model_path
    global num_string_model_path
    # global point_string_model_path
    global sign_string_model_path
    num_string_model_path, sign_string_model_path, eng_string_model_path, chi_string_model_path = (
     num_model_path_val, sign_model_path_val, eng_model_path_val, chi_model_path_val)


def crnn(model_path, cropped_image, model):
    if model_path == num_string_model_path:
        converter = cvt.strLabelConverter(num_alphabet)
    elif model_path == sign_string_model_path:
        converter = cvt.strLabelConverter(sign_alphabet)
    elif model_path == eng_string_model_path:
        converter = cvt.strLabelConverter(eng_alphabet)
    elif model_path == chi_string_model_path:
        converter = cvt.strLabelConverter(chi_alphabet)
    # elif model_path == point_string_model_path:
    #     converter = cvt.strLabelConverter(point_alphabet)
    # elif model_path == douhao_string_model_path:
    #     converter = cvt.strLabelConverter(douhao_alphabet)
    image = cropped_image.convert('L')
    w = image.size[0]
    transformer = dt.resizeNormalize((w, conf.img_height))
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = (image.view)(*(1, ), *image.size())
    image = Variable(image)
    model.eval()
    preds = model(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode((preds.data), (preds_size.data), raw=False)
    return sim_pred


def recognize(running_mode, model_path, img_path, result_path):
    if model_path == num_string_model_path:
        model = Net.CRNN(num_class)
        crnn_model_path = num_string_model_path
        IMG_ROOT = img_path
    elif model_path == sign_string_model_path:
        model = Net.CRNN(sign_class)
        crnn_model_path = sign_string_model_path
        IMG_ROOT = img_path
    elif model_path == eng_string_model_path:
        model = Net.CRNN(eng_class)
        crnn_model_path = eng_string_model_path
        IMG_ROOT = img_path
    elif model_path == chi_string_model_path:
        model = Net.CRNN(chi_class)
        crnn_model_path = chi_string_model_path
        IMG_ROOT = img_path
    # elif model_path == point_string_model_path:
    #     model = Net.CRNN(point_class)
    #     crnn_model_path = point_string_model_path
    #     IMG_ROOT = img_path
    # elif model_path == douhao_string_model_path:
    #     model = Net.CRNN(douhao_class)
    #     crnn_model_path = douhao_string_model_path
    #     IMG_ROOT = img_path
    if running_mode == 'gpu':
        if torch.cuda.is_available():
            model = model.cuda()
            model.load_state_dict((torch.load(crnn_model_path)), strict=False)
        model.load_state_dict(torch.load(crnn_model_path, map_location='cpu'))
    print('loading pretrained model from {0}'.format(crnn_model_path))
    s1 = []
    s2 = []
    df = pd.DataFrame()
    files = sorted(os.listdir(IMG_ROOT))
    for file in files:
        started = time.time()
        full_path = os.path.join(IMG_ROOT, file)
        # print('=============================================')
        # print('ocr image is %s' % full_path)
        index = file[:-4].split('_')[1]
        s1.extend([index])
        image = Image.open(full_path)
        value = crnn(model_path, image, model)
        if len(value) == 0:
            value = 'None'
        s2.extend([value])
        finished = time.time()
        # print('elapsed time: {0}'.format(finished - started))

    df['index'] = s1
    df['value'] = s2
    df.set_index('index', drop=True, inplace=True)
    return df
# okay decompiling string_char_def.pyc
