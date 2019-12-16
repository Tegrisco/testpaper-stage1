# uncompyle6 version 3.5.1
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.7.4 (default, Sep  7 2019, 18:27:02) 
# [Clang 10.0.1 (clang-1001.0.46.4)]
# Embedded file name: Config.py
# Compiled at: 2019-09-12 11:47:01
# Size of source mod 2**32: 966 bytes
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import howso.num_alphabets as numalpha, howso.sign_alphabets as signalpha, howso.eng_alphabets as engalpha, howso.chi_alphabets as chialpha, howso.point_alphabets as pointalpha, howso.douhao_alphabets as douhaoalpha
random_sample = True
random_seed = 200
using_cuda = True
keep_ratio = True
gpu_id = '0,1'
model_dir = os.path.join(BASE_DIR, '../oral/num_string_model')
data_worker = 2
batch_size = 256
img_height = 32
img_width = 128
num_alphabet = numalpha.alphabet
sign_alphabet = signalpha.alphabet
eng_alphabet = engalpha.alphabet
chi_alphabet = chialpha.alphabet
point_alphabet = pointalpha.alphabet
douhao_alphabet = douhaoalpha.alphabet
epoch = 61
display_interval = 20
save_interval = 2000
test_interval = 1000
test_disp = 50
test_batch_num = 256
lr = 0.0001
beta1 = 0.5
infer_img_w = 64
# okay decompiling Config.pyc
