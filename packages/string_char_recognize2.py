import howso.string_char_def as ssr
import sys

#初始化参数
num_string_model_path = '../oral/num_string_model/num_string_model'
sign_string_model_path = '../oral/sign_string_model/sign_string_model'
eng_string_model_path = '../oral/eng_string_model/eng_string_model'
chi_string_model_path = '../oral/chi_string_model/chi_string_model'
save_path = '../oral/csv'
running_mode = 'cpu'

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        timeN = sys.argv[1]
    else:
        print("no valid input argument, exit...")
        sys.exit()

    num_string_img_path = '../oral/img_store/'+ timeN +'num_string_img/'
    sign_string_img_path = '../oral/img_store/'+ timeN +'sign_string_img/'
    eng_string_img_path = '../oral/img_store/'+ timeN +'eng_string_img/'
    chi_strin_img_path = '../oral/img_store/'+ timeN +'chi_string_img'

    ssr.init(num_string_model_path,sign_string_model_path,eng_string_model_path,chi_string_model_path)
    
    print('========start number string recognize============')
    num_string_df = ssr.recognize(running_mode,num_string_model_path,num_string_img_path,save_path)
   # print('========start sign string recognize============')
   # sign_string_df = ssr.recognize(running_mode,sign_string_img_path,sign_string_img_path,save_path)
   # print('========start english string recognize============')
   # eng_string_df = ssr.recognize(running_mode,eng_string_model_path,eng_string_img_path,save_path)
    print('========start chinese string recognize============')
    chi_string_df = ssr.recognize(running_mode,chi_string_model_path,chi_strin_img_path,save_path)
   
    # chi_string_df.to_csv(save_path + '/'+timeN +'chi_string_ans_result.csv')
   # eng_string_df.to_csv(save_path + '/'+'eng_string_ans_result.csv')
    num_string_df.to_csv(save_path + '/'+timeN +'num_string_ans_result.csv')
   # sign_string_df.to_csv(save_path + '/'+'sign_string_ans_result.csv')
