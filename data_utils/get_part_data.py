import glob
from utils import shuffle_file, distribution_file

ori_train_img_file_path = '../data/train/img/'
ori_train_label_file_path = '../data/train/label/'
ori_val_img_file_path = '../data/val/img/'
ori_val_label_file_path = '../data/val/label/'
ori_test_img_file_path = '../data/test/img/'
ori_test_label_file_path = '../data/test/label/'

part_train_img_file_path = '../data/part_data/train/img/'
part_train_label_file_path = '../data/part_data/train/label/'
part_val_img_file_path = '../data/part_data/val/img/'
part_val_label_file_path = '../data/part_data/val/label/'
part_test_img_file_path = '../data/part_data/test/img/'
part_test_label_file_path = '../data/part_data/test/label/'

ori_train_img_file_list = glob.glob(ori_train_img_file_path + '*.jpg')
ori_train_label_file_list = glob.glob(ori_train_label_file_path + '*.png')
ori_val_img_file_list = glob.glob(ori_val_img_file_path + '*.jpg')
ori_val_label_file_list = glob.glob(ori_val_label_file_path + '*.png')
ori_test_img_file_list = glob.glob(ori_test_img_file_path + '*.jpg')
ori_test_label_file_list = glob.glob(ori_test_label_file_path + '*.png')

shuffle_train_img_file_list, shuffle_train_label_file_list = shuffle_file(ori_train_img_file_list, ori_train_label_file_list)
shuffle_val_img_file_list, shuffle_val_label_file_list = shuffle_file(ori_val_img_file_list, ori_val_label_file_list)
shuffle_test_img_file_list, shuffle_test_label_file_list = shuffle_file(ori_test_img_file_list, ori_test_label_file_list)

distribution_file(shuffle_train_img_file_list[:5000], shuffle_train_label_file_list[:5000],
                  part_train_img_file_path, part_train_label_file_path)
distribution_file(shuffle_val_img_file_list[:500], shuffle_val_label_file_list[:500],
                  part_val_img_file_path, part_val_label_file_path)
distribution_file(shuffle_test_img_file_list[:500], shuffle_test_label_file_list[:500],
                  part_test_img_file_path, part_test_label_file_path)