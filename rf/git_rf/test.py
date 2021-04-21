import scipy.io as scio
import scipy.misc as misc
import numpy as np
import h5py
import os
from sklearn.ensemble import RandomForestClassifier
import math
import time
import cv2
import pickle
import matplotlib.pyplot as pyplot

from utils import utils


def test_process_one_image(root_path_x, root_path_y, img_path):
    # data = h5py.File(filepath, 'r')
    # current_model_path = '../python/model/'
    # current_model_name = 'rf_clf_50.pickle'

    current_model_path = './model/'
    # current_model_path = '../python/model/'
    # current_model_name = 'rf_clf_512_50all_100.pickle'
    # current_model_name = 'rf_clf_512_40all_5.pickle'
    # current_model_name = 'rf_clf_100all_100_cnn3.pickle'
    current_model_name = 'rf_clf_512_40all_100_cnn512_rgb_xy.pickle'

    with open(current_model_path + current_model_name, 'rb') as f:
        rf = pickle.load(f)

    for file in os.listdir(root_path_x):

        x_file_path = root_path_x + "/"+ file
        y_file_path = root_path_y + "/" + file

        print('x_file_path:', x_file_path)
        print('y_file_path:', y_file_path)

        # get hks deep_featrue x_train
        xy, rgb, hks, deep_featrue, points, x_test = utils.load_x_file(x_file_path, is_norm=False)

        result_image = np.zeros((700, 700))
        y_test = utils.load_y_file(y_file_path)


        y_pred_ = rf.predict(x_test)

        for i in range(0, len(y_pred_)):
            # print(y_pred_[i])
            if(y_pred_[i ] >0):
                for p in points[i]:
                    result_image[p[0]][p[1] ] =y_pred_[i]

        # for i in range(0,len(result_image)):
        #    print(result_image[i])
        filename = file.split('.')
        img_p_name = img_path +'/' +filename[0 ] +'.png'
        cv2.imwrite(img_p_name, result_image)
        # misc.imsave(img_p_name,result_image)

        pyplot.imshow(result_image)
        # pyplot.savefig(img_path+'/'+filename[0]+'.png')
        pyplot.show()


if __name__ == '__main__':


    # test_path_x_ = '/home/pzn/pzncode/yjl/to_yuan/datasets/cnn3_small/cnn3_train'
    # test_path_y_ = '/home/pzn/pzncode/yjl/to_yuan/datasets/cnn3_small/label'
    # img_path = "/home/pzn/pzncode/yjl/to_yuan/datasets/cnn3_small/result"

    # test_path_x_ = '/home/pzn/pzncode/yjl/to_yuan/datasets/cnn3/JPEGImages_3_test'
    # test_path_y_ = '/home/pzn/pzncode/yjl/to_yuan/datasets/cnn3/all_label_train_new_version'

    test_path_x_ = '/home/pzn/pzncode/yjl/to_yuan/datasets/cnn512/JPEGImages_512_val'
    test_path_y_ = '/home/pzn/pzncode/yjl/to_yuan/datasets/cnn3/all_label_train_new_version'

    img_path = "./result_512_40all_100_cnn512_rgb_xy_val"
    if not os.path.isdir(img_path):
        os.mkdir(img_path)
    test_process_one_image(test_path_x_, test_path_y_, img_path)
    # model = compute_params(x_train_, one_y)
    localtime = time.asctime(time.localtime(time.time()))
    print("Finished Time:", localtime)

