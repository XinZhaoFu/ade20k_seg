from utils import shuffle_file, distribution_file, recreate_dir
import glob

ori_img_file_path = '../data/ori-img/'
ori_label_file_path = '../data/ori_annotation/'
dis_img_train_file_path = '../data/train/img/'
dis_label_train_file_path = '../data/train/label/'
dis_img_val_file_path = '../data/val/img/'
dis_label_val_file_path = '../data/val/label/'
dis_img_test_file_path = '../data/test/img/'
dis_label_test_file_path = '../data/test/label/'

ori_img_file_list = glob.glob(ori_img_file_path + '*.jpg')
ori_label_file_list = glob.glob(ori_label_file_path + '*.png')
assert len(ori_img_file_list) == len(ori_label_file_list)
shuffle_img_file_list, shuffle_label_file_list = shuffle_file(ori_img_file_list, ori_label_file_list)
# for ori_img_file, ori_label_file in zip(shuffle_img_file_list, shuffle_label_file_list):
#     print(ori_img_file, ori_label_file)

recreate_dir(dis_img_train_file_path)
recreate_dir(dis_label_train_file_path)
recreate_dir(dis_img_val_file_path)
recreate_dir(dis_label_val_file_path)
recreate_dir(dis_img_test_file_path)
recreate_dir(dis_label_test_file_path)

distribution_file(shuffle_img_file_list[:int(len(ori_img_file_list) * 0.8)],
                  shuffle_label_file_list[:int(len(ori_img_file_list) * 0.8)],
                  dis_img_train_file_path, dis_label_train_file_path)

distribution_file(shuffle_img_file_list[int(len(ori_img_file_list) * 0.8):int(len(ori_img_file_list) * 0.9)],
                  shuffle_label_file_list[int(len(ori_img_file_list) * 0.8):int(len(ori_img_file_list) * 0.9)],
                  dis_img_val_file_path, dis_label_val_file_path)

distribution_file(shuffle_img_file_list[int(len(ori_img_file_list) * 0.9):],
                  shuffle_label_file_list[int(len(ori_img_file_list) * 0.9):],
                  dis_img_test_file_path, dis_label_test_file_path)
