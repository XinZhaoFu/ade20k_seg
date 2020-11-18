import glob

ori_img_file_path = './data/ori-img/'
ori_label_file_path = './data/ori-label/'

ori_img_file_list = glob.glob(ori_img_file_path + '*.jpg')
ori_label_file_list = glob.glob(ori_label_file_path + '*.png')

print(len(ori_img_file_list), len(ori_label_file_list))