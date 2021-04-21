# *_*coding:utf-8 *_*
# author: JoannaPang
# 经典LBP算法复现：原始LBP、Uniform LBP、旋转不变LBP、旋转不变的Uniform LBP

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os

def LBP(src):
    '''
    :param src:灰度图像
    :return:
    '''
    height = src.shape[0]
    width = src.shape[1]
    # dst = np.zeros([height, width], dtype=np.uint8)
    dst = src.copy()

    lbp_value = np.zeros((1,8), dtype=np.uint8)
    neighbours = np.zeros((1,8), dtype=np.uint8)
    for x in range(1, width-1):
        for y in range(1, height-1):
            neighbours[0, 0] = src[y - 1, x - 1]
            neighbours[0, 1] = src[y - 1, x]
            neighbours[0, 2] = src[y - 1, x + 1]
            neighbours[0, 3] = src[y, x - 1]
            neighbours[0, 4] = src[y, x + 1]
            neighbours[0, 5] = src[y + 1, x - 1]
            neighbours[0, 6] = src[y + 1, x]
            neighbours[0, 7] = src[y + 1, x + 1]

            center = src[y, x]

            for i in range(8):
                if neighbours[0, i] > center:
                    lbp_value[0, i] = 1
                else:
                    lbp_value[0, i] = 0

            lbp = lbp_value[0, 0] * 1 + lbp_value[0, 1] * 2 + lbp_value[0, 2] * 4 + lbp_value[0, 3] * 8 \
                + lbp_value[0, 4] * 16 + lbp_value[0, 5] * 32 + lbp_value[0, 6] * 64 + lbp_value[0, 0] * 128

            dst[y, x] = lbp

    return dst

def getHopCnt(num):
    '''
    :param num:8位的整形数，0-255
    :return:
    '''
    if num > 255:
        num = 255
    elif num < 0:
        num = 0

    num_b = bin(num)
    num_b = str(num_b)[2:]

    # 补0
    if len(num_b) < 8:
        temp = []
        for i in range(8-len(num_b)):
            temp.append('0')
        temp.extend(num_b)
        num_b = temp

    cnt = 0
    for i in range(8):
        if i == 0:
            former = num_b[-1]
        else:
            former = num_b[i-1]
        if former == num_b[i]:
            pass
        else:
            cnt += 1

    return cnt

def uniform_LBP(src, norm=True):
    '''
    :param src:原始图像
    :param norm:是否做归一化到【0-255】的灰度空间
    :return:
    '''
    table = np.zeros((256), dtype=np.uint8)
    temp = 1
    for i in range(256):
        if getHopCnt(i) <= 2:
            table[i] = temp
            temp += 1

    height = src.shape[0]
    width = src.shape[1]
    dst = np.zeros([height, width], dtype=np.uint8)
    dst = src.copy()

    lbp_value = np.zeros((1, 8), dtype=np.uint8)
    neighbours = np.zeros((1, 8), dtype=np.uint8)
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            neighbours[0, 0] = src[y - 1, x - 1]
            neighbours[0, 1] = src[y - 1, x]
            neighbours[0, 2] = src[y - 1, x + 1]
            neighbours[0, 3] = src[y, x - 1]
            neighbours[0, 4] = src[y, x + 1]
            neighbours[0, 5] = src[y + 1, x - 1]
            neighbours[0, 6] = src[y + 1, x]
            neighbours[0, 7] = src[y + 1, x + 1]

            center = src[y, x]

            for i in range(8):
                if neighbours[0, i] > center:
                    lbp_value[0, i] = 1
                else:
                    lbp_value[0, i] = 0

            lbp = lbp_value[0, 0] * 1 + lbp_value[0, 1] * 2 + lbp_value[0, 2] * 4 + lbp_value[0, 3] * 8 \
                  + lbp_value[0, 4] * 16 + lbp_value[0, 5] * 32 + lbp_value[0, 6] * 64 + lbp_value[0, 0] * 128

            dst[y, x] = table[lbp]

    if norm is True:
        return img_max_min_normalization(dst)
    else:
        return dst

def img_max_min_normalization(src, min=0, max=255):
    height = src.shape[0]
    width = src.shape[1]
    if len(src.shape) > 2:
        channel = src.shape[2]
    else:
        channel = 1

    src_min = np.min(src)
    src_max = np.max(src)

    if channel == 1:
        dst = np.zeros([height, width], dtype=np.float32)
        for h in range(height):
            for w in range(width):
                dst[h, w] = float(src[h, w] - src_min) / float(src_max - src_min) * (max - min) + min
    else:
        dst = np.zeros([height, width, channel], dtype=np.float32)
        for c in range(channel):
            for h in range(height):
                for w in range(width):
                    dst[h, w, c] = float(src[h, w, c] - src_min) / float(src_max - src_min) * (max - min) + min

    return dst

def value_rotation(num):
    value_list = np.zeros((8), np.uint8)
    temp = int(num)
    value_list[0] = temp
    for i in range(7):
        #temp = ((temp << 1) | (temp / 128)) % 256
        temp_fenmu = int(temp / 128)
        temp = ((temp << 1) | temp_fenmu) % 256
        value_list[i+1] = temp
    return np.min(value_list)

def rotation_invariant_LBP(src):
    height = src.shape[0]
    width = src.shape[1]
    # dst = np.zeros([height, width], dtype=np.uint8)
    dst = src.copy()

    lbp_value = np.zeros((1, 8), dtype=np.uint8)
    neighbours = np.zeros((1, 8), dtype=np.uint8)
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            neighbours[0, 0] = src[y - 1, x - 1]
            neighbours[0, 1] = src[y - 1, x]
            neighbours[0, 2] = src[y - 1, x + 1]
            neighbours[0, 3] = src[y, x - 1]
            neighbours[0, 4] = src[y, x + 1]
            neighbours[0, 5] = src[y + 1, x - 1]
            neighbours[0, 6] = src[y + 1, x]
            neighbours[0, 7] = src[y + 1, x + 1]

            center = src[y, x]

            for i in range(8):
                if neighbours[0, i] > center:
                    lbp_value[0, i] = 1
                else:
                    lbp_value[0, i] = 0

            lbp = lbp_value[0, 0] * 1 + lbp_value[0, 1] * 2 + lbp_value[0, 2] * 4 + lbp_value[0, 3] * 8 \
                  + lbp_value[0, 4] * 16 + lbp_value[0, 5] * 32 + lbp_value[0, 6] * 64 + lbp_value[0, 0] * 128

            dst[y, x] = value_rotation(lbp)

    return dst

def rotation_invariant_uniform_LBP(src):
    table = np.zeros((256), dtype=np.uint8)
    temp = 1
    for i in range(256):
        if getHopCnt(i) <= 2:
            table[i] = temp
            temp += 1

    height = src.shape[0]
    width = src.shape[1]
    dst = np.zeros([height, width], dtype=np.uint8)
    dst = src.copy()

    lbp_value = np.zeros((1, 8), dtype=np.uint8)
    neighbours = np.zeros((1, 8), dtype=np.uint8)
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            neighbours[0, 0] = src[y - 1, x - 1]
            neighbours[0, 1] = src[y - 1, x]
            neighbours[0, 2] = src[y - 1, x + 1]
            neighbours[0, 3] = src[y, x - 1]
            neighbours[0, 4] = src[y, x + 1]
            neighbours[0, 5] = src[y + 1, x - 1]
            neighbours[0, 6] = src[y + 1, x]
            neighbours[0, 7] = src[y + 1, x + 1]

            center = src[y, x]

            for i in range(8):
                if neighbours[0, i] > center:
                    lbp_value[0, i] = 1
                else:
                    lbp_value[0, i] = 0

            lbp = lbp_value[0, 0] * 1 + lbp_value[0, 1] * 2 + lbp_value[0, 2] * 4 + lbp_value[0, 3] * 8 \
                  + lbp_value[0, 4] * 16 + lbp_value[0, 5] * 32 + lbp_value[0, 6] * 64 + lbp_value[0, 0] * 128

            dst[y, x] = table[lbp]

    dst = img_max_min_normalization(dst)
    for x in range(width):
        for y in range(height):
            dst[y, x] = value_rotation(dst[y, x])

    return dst

def circular_LBP(src, radius, n_points):
    height = src.shape[0]
    width = src.shape[1]
    print('height: ',height)
    print('width: ',width)
    # dst = np.zeros([height, width], dtype=np.uint8)
    dst = src.copy()
    src.astype(dtype=np.float32)
    dst.astype(dtype=np.float32)

    neighbours = np.zeros((1, n_points), dtype=np.uint8)
    lbp_value = np.zeros((1, n_points), dtype=np.uint8)
    for x in range(radius, width - radius - 1):
        for y in range(radius, height - radius - 1):
            lbp = 0.
            for n in range(n_points):
                theta = float(2 * np.pi * n) / n_points
                x_n = x + radius * np.cos(theta)
                y_n = y - radius * np.sin(theta)

                # 向下取整
                x1 = int(math.floor(x_n))
                y1 = int(math.floor(y_n))
                # 向上取整
                x2 = int(math.ceil(x_n))
                y2 = int(math.ceil(y_n))

                # 将坐标映射到0-1之间
                tx = np.abs(x - x1)
                ty = np.abs(y - y1)

                # 根据0-1之间的x，y的权重计算公式计算权重
                w1 = (1 - tx) * (1 - ty)
                w2 = tx * (1 - ty)
                w3 = (1 - tx) * ty
                w4 = tx * ty

                # 根据双线性插值公式计算第k个采样点的灰度值
                neighbour = src[y1, x1] * w1 + src[y2, x1] * w2 + src[y1, x2] * w3 + src[y2, x2] * w4

                neighbours[0, n] = neighbour

            center = src[y, x]

            # print('center:{}; neighbours:{}'.format(center, neighbours))

            for n in range(n_points):
                if neighbours[0, n] > center:
                    lbp_value[0, n] = 1
                else:
                    lbp_value[0, n] = 0

            # print('lbp_value:{}'.format(lbp_value))

            for n in range(n_points):
                lbp += lbp_value[0, n] * 2**n
                # print('lbp_value[0, n] * 2**n : {}'.format(lbp_value[0, n] * 2**n))

            # print('lbp_value transformed:{}'.format(lbp))

            dst[y, x] = int(lbp / (2**n_points-1) * 255)

            # print('dst value of [{}, {}]:{}'.format(y, x, dst[y,x]))

    return dst

def disp_test_result(img, gray, dst, path, name, mode=0):
    '''
    :param mode:0,opencv显示图片；1,matplotlib显示图片。
    :return:
    '''
    if mode == 0:
        cv2.imshow('src', img)
        cv2.imshow('gray', gray)
        cv2.imshow('LBP', dst)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        plt.figure()
        plt.subplot(131)
        plt.imshow(img)
        plt.title('src')

        plt.subplot(132)
        plt.imshow(gray, cmap='gray')
        plt.title('gray')

        plt.subplot(133)
        plt.imshow(dst, cmap='gray')
        plt.title('LBP')
        path_name = path + '/' + name
        print('The all result path is:',path_name)
        plt.savefig(path_name)######xu!!!???

        #plt.show()

def init_dir(rootdst,graydst,dst,dst1,dst2,dst3,dst4,all_dst):
    

    if not os.path.exists(rootdst):
        os.makedirs(rootdst)
        print('making rootdst')
    else:
        print('already maked rootdst')

    if not os.path.exists(graydst):
        os.makedirs(graydst)
        print('making graydst')
    else:
        print('already maked graydst')

    if not os.path.exists(dst):
        os.makedirs(dst)
        print('making dst')
    else:
        print('already maked dst')

    if not os.path.exists(dst1):
        os.makedirs(dst1)
        print('making dst1')
    else:
        print('already maked dst1')

    if not os.path.exists(dst2):
        os.makedirs(dst2)
        print('making dst2')
    else:
        print('already maked dst2')

    if not os.path.exists(dst3):
        os.makedirs(dst3)
        print('making dst3')
    else:
        print('already maked dst3')

    if not os.path.exists(dst4):
        os.makedirs(dst4)
        print('making dst4')
    else:
        print('already maked dst4')

    if not os.path.exists(all_dst):
        os.makedirs(all_dst)
        print('making all_dst')
    else:
        print('already maked all_dst')


if __name__ == '__main__':
    
    rootdir = os.path.abspath('./JPEGImages_new/')#image folder

    root_path = os.path.abspath('./src_src/')#yuan image
    graydst_path = os.path.abspath('./gray/')#gray image
    dst_path = os.path.abspath('./dst/')#dst image
    dst1_path = os.path.abspath('./dst1/')#dst1 image
    dst2_path = os.path.abspath('./dst2/')#dst2 image
    dst3_path = os.path.abspath('./dst3/')#dst3 image
    dst4_path = os.path.abspath('./dst4/')#dst4 image
    all_dst_path = os.path.abspath('./all_dst/')

    init_dir(root_path,graydst_path,dst_path,dst1_path,dst2_path,dst3_path,dst4_path,all_dst_path)

    imglist = os.listdir(rootdir)
    imglist.sort()

    print(len(imglist))
    
    for i in range(20,len(imglist)):
    #for i in range(0,len(imglist)):
    #for i in range(405,len(imglist)):
    #for i in range(0,1):
        print('The image name is:',imglist[i])
        front_name, back_name = os.path.splitext(imglist[i])
        print('The front_name is:',front_name)
        print('The back_name is:',back_name)
        imgpath = os.path.join(rootdir,imglist[i])
        img = cv2.imread(imgpath)
    
    
        print('get img')
        #cv2.imwrite('./dst2/img2.jpg',img)
        #cv2.imwrite('./dst2/img1.jpg',img, [int(cv2.IMWRITE_JPEG_QUALITY),5])
        root_path_name = root_path + '/' + imglist[i]
        cv2.imwrite(root_path_name, img, [int(cv2.IMWRITE_JPEG_QUALITY),100])
        
    

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print('get gray')
        graydst_path_name = graydst_path + '/' + front_name + '.png'
        cv2.imwrite(graydst_path_name, gray, [int(cv2.IMWRITE_PNG_COMPRESSION),0])

        '''
        dst = LBP(gray)
        print('get dst')
        dst_path_name = dst_path + '/' + front_name + '.png'
        cv2.imwrite(dst_path_name, dst, [int(cv2.IMWRITE_PNG_COMPRESSION),0])


        dst1 = uniform_LBP(gray)
        print('get dst1')
        dst1_path_name = dst1_path + '/' + front_name + '.png'
        cv2.imwrite(dst1_path_name, dst1, [int(cv2.IMWRITE_PNG_COMPRESSION),0])


        dst2 = rotation_invariant_LBP(gray)
        print('get dst2')
        dst2_path_name = dst2_path + '/' + front_name + '.png'
        cv2.imwrite(dst2_path_name, dst2, [int(cv2.IMWRITE_PNG_COMPRESSION),0])


        dst3 = rotation_invariant_uniform_LBP(gray)
        print('get dst3')
        dst3_path_name = dst3_path + '/' + front_name + '.png'
        cv2.imwrite(dst3_path_name, dst3, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
        '''

        dst4 = circular_LBP(gray, radius=4, n_points=16)
        print('get dst4')
        dst4_path_name = dst4_path + '/' + front_name + '.png'
        cv2.imwrite(dst4_path_name, dst4, [int(cv2.IMWRITE_PNG_COMPRESSION),0])



        #disp_test_result(img, gray, dst, all_dst_path, front_name + '_dst.png', mode=1)
        #disp_test_result(img, gray, dst1, all_dst_path, front_name + '_dst1.png', mode=1)
        #disp_test_result(img, gray, dst2, all_dst_path, front_name + '_dst2.png', mode=1)
        #disp_test_result(img, gray, dst3, all_dst_path, front_name + '_dst3.png', mode=1)
        disp_test_result(img, gray, dst4, all_dst_path, front_name + '_dst4.png', mode=1)

    print(type(img))
    print(img.shape)
    print(type(gray))
    print(gray.shape)
    '''
    print(type(dst))
    print(dst.shape)
    print(type(dst1))
    print(dst1.shape)
    print(type(dst2))
    print(dst2.shape)
    print(type(dst3))
    print(dst3.shape)
    '''
    print(type(dst4))
    print(dst4.shape)
    
    



#conda activate lbp

