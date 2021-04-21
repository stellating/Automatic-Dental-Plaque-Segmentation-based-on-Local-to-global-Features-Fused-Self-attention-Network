import sys
import os
import shutil


train_text_path = './ImageSets/Segmentation/train.txt'
val_text_path = './ImageSets/Segmentation/val.txt'
test_text_path = './ImageSets/Segmentation/test.txt'
old_file_path = './deep_3_hks_lbp_as_z_result_20201104_slim_high/'
train_file_path = './JPEGImages_3_train/'
val_file_path = './JPEGImages_3_val/'
test_file_path = './JPEGImages_3_test/'

if not os.path.exists(train_file_path):
    os.mkdir(train_file_path)
if not os.path.exists(val_file_path):
    os.mkdir(val_file_path)
if not os.path.exists(test_file_path):
    os.mkdir(test_file_path)

train_file = open(train_text_path)
for line in train_file:
    # print('line:', line)
    image_old_path = old_file_path + line.strip() + '.mat'
    print('image_old_path:', image_old_path)
    image_new_path = train_file_path + line.strip() + '.mat'
    shutil.copyfile(image_old_path, image_new_path)
train_file.close()

val_file = open(val_text_path)
for line in val_file:
    # print('line:', line)
    image_old_path = old_file_path + line.strip() + '.mat'
    print('image_old_path:', image_old_path)
    image_new_path = val_file_path + line.strip() + '.mat'
    shutil.copyfile(image_old_path, image_new_path)
val_file.close()

test_file = open(test_text_path)
for line in test_file:
    # print('line:', line)
    image_old_path = old_file_path + line.strip() + '.mat'
    print('image_old_path:', image_old_path)
    image_new_path = test_file_path + line.strip() + '.mat'
    shutil.copyfile(image_old_path, image_new_path)
test_file.close()

print('raw images finished!')

'''
old_file_path = '../SegmentationClass/'
train_file_path = '../SegmentationClass_train/'
val_file_path = '../SegmentationClass_val/'
test_file_path = '../SegmentationClass_test/'


if not os.path.exists(train_file_path):
    os.mkdir(train_file_path)
if not os.path.exists(val_file_path):
    os.mkdir(val_file_path)
if not os.path.exists(test_file_path):
    os.mkdir(test_file_path)


train_file = open(train_text_path)
for line in train_file:
    # print('line:', line)
    image_old_path = old_file_path + line.strip() + '.png'
    print('image_old_path:', image_old_path)
    image_new_path = train_file_path + line.strip() + '.png'
    shutil.copyfile(image_old_path, image_new_path)
train_file.close()


val_file = open(val_text_path)
for line in val_file:
    # print('line:', line)
    image_old_path = old_file_path + line.strip() + '.png'
    print('image_old_path:', image_old_path)
    image_new_path = val_file_path + line.strip() + '.png'
    shutil.copyfile(image_old_path, image_new_path)
val_file.close()


test_file = open(test_text_path)
for line in test_file:
    # print('line:', line)
    image_old_path = old_file_path + line.strip() + '.png'
    print('image_old_path:', image_old_path)
    image_new_path = test_file_path + line.strip() + '.png'
    shutil.copyfile(image_old_path, image_new_path)
test_file.close()
'''
