# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:13:49 2019

@author: howso
"""

import sys
import howso.sigle_char_def as sr

#初始化参数
chi_dict = '../oral/dict/chi_dict' #中文字典
eng_dict = '../oral/dict/eng_dict' #英文字典
num_dict = '../oral/dict/num_dict' #数字字典
sign_dict = '../oral/dict/sign_dict' #中文字典
save_path = '../oral/csv' #答案识别结果保存路径
if __name__=='__main__':
    if len(sys.argv) >= 2:
        timeN = sys.argv[1]
    else:
        print("no valid input argument, exit...")
        sys.exit()

    chi_img_path = '../oral/img_store/'+ timeN +'chi_img'
    eng_img_path = '../oral/img_store/'+ timeN +'eng_img'
    num_img_path = '../oral/img_store/'+ timeN +'num_img'
    sign_img_path = '../oral/img_store/'+ timeN +'sign_img'

    sr.init()
    sr.dict_val(chi_dict,eng_dict,num_dict,sign_dict)
    print('========start chinese recognize============')
    chi_df = sr.recognize(chi_dict,chi_img_path,save_path)
    print('========start english recognize============')
    eng_df = sr.recognize(eng_dict,eng_img_path,save_path)
    print('========start number recognize============')
    num_df = sr.recognize(num_dict,num_img_path,save_path)
    print('========start sign recognize============')
    sign_df = sr.recognize(sign_dict,sign_img_path,save_path)
    chi_df.to_csv(save_path + '/'+ timeN +'chi_ans_result.csv')
    eng_df.to_csv(save_path + '/'+ timeN +'eng_ans_result.csv')
    num_df.to_csv(save_path + '/'+ timeN +'num_ans_result.csv')
    sign_df.to_csv(save_path + '/'+ timeN +'sign_ans_result.csv')
    sr.vals_clr()
