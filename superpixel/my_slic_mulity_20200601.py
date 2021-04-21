import math
from skimage import io, color
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import scipy.io as scio
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Cluster(object):
    cluster_index = 1

    def __init__(self, h, w, l=0, a=0, b=0):
        self.update(h, w, l, a, b)
        self.pixels = []
        self.no = self.cluster_index
        Cluster.cluster_index += 1

    def update(self, h, w, l, a, b):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b

    def __str__(self):
        return "{},{}:{} {} {} ".format(self.h, self.w, self.l, self.a, self.b)

    def __repr__(self):
        return self.__str__()

class SLICProcessor(object):
    @staticmethod
    def open_image(path):
        """
        Return:
            3D array, row col [LAB]
        """
        rgb = io.imread(path)
        lab_arr = color.rgb2lab(rgb)
        return lab_arr

    @staticmethod
    def save_lab_image(path, lab_arr):
        """
        Convert the array to RBG, then save the image
        :param path:
        :param lab_arr:
        :return:
        """
        rgb_arr = color.lab2rgb(lab_arr)
        io.imsave(path, rgb_arr)

    def make_cluster(self, h, w):
        #print('type of h:',type(h))#pang_add_20190709
        #print('type of w:',type(w))#pang_add_20190709
        h = int(h)#pang_add_20190709
        w = int(w)#pang_add_20190709
        return Cluster(h, w,
                       self.data[h][w][0],
                       self.data[h][w][1],
                       self.data[h][w][2])

    def __init__(self, filename, K, M):
        self.K = K
        self.M = M

        self.data = self.open_image(filename)
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.N = self.image_height * self.image_width
        self.S = int(math.sqrt(self.N / self.K))

        self.clusters = []
        self.label = {}
        self.dis = np.full((self.image_height, self.image_width), np.inf)

    def init_clusters(self):
        h = self.S / 2
        w = self.S / 2
        while h < self.image_height:
            while w < self.image_width:
                self.clusters.append(self.make_cluster(h, w))
                w += self.S
            w = self.S / 2
            h += self.S

    def get_gradient(self, h, w):
        if w + 1 >= self.image_width:
            w = self.image_width - 2
        if h + 1 >= self.image_height:
            h = self.image_height - 2

        gradient = self.data[w + 1][h + 1][0] - self.data[w][h][0] + \
                   self.data[w + 1][h + 1][1] - self.data[w][h][1] + \
                   self.data[w + 1][h + 1][2] - self.data[w][h][2]
        return gradient

    def move_clusters(self):
        for cluster in self.clusters:
            cluster_gradient = self.get_gradient(cluster.h, cluster.w)
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    _h = cluster.h + dh
                    _w = cluster.w + dw
                    new_gradient = self.get_gradient(_h, _w)
                    if new_gradient < cluster_gradient:
                        cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])
                        cluster_gradient = new_gradient

    def assignment(self):
        for cluster in self.clusters:
            for h in range(cluster.h - 2 * self.S, cluster.h + 2 * self.S):
                if h < 0 or h >= self.image_height: continue
                for w in range(cluster.w - 2 * self.S, cluster.w + 2 * self.S):
                    if w < 0 or w >= self.image_width: continue
                    L, A, B = self.data[h][w]
                    Dc = math.sqrt(
                        math.pow(L - cluster.l, 2) +
                        math.pow(A - cluster.a, 2) +
                        math.pow(B - cluster.b, 2))
                    Ds = math.sqrt(
                        math.pow(h - cluster.h, 2) +
                        math.pow(w - cluster.w, 2))
                    D = math.sqrt(math.pow(Dc / self.M, 2) + math.pow(Ds / self.S, 2))
                    if D < self.dis[h][w]:
                        if (h, w) not in self.label:
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        else:
                            self.label[(h, w)].pixels.remove((h, w))
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        self.dis[h][w] = D

    def update_cluster(self):
        for cluster in self.clusters:
            sum_h = sum_w = number = 0
            for p in cluster.pixels:
                sum_h += p[0]
                sum_w += p[1]
                number += 1
                _h = sum_h / number
                _w = sum_w / number
                _h = int(_h)#pang_add_20190709
                _w = int(_w)#pang_add_20190709
                cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])

    def save_current_image(self, name):
        image_arr = np.copy(self.data)
        for cluster in self.clusters:
            for p in cluster.pixels:
                image_arr[p[0]][p[1]][0] = cluster.l
                image_arr[p[0]][p[1]][1] = cluster.a
                image_arr[p[0]][p[1]][2] = cluster.b
            image_arr[cluster.h][cluster.w][0] = 0
            image_arr[cluster.h][cluster.w][1] = 0
            image_arr[cluster.h][cluster.w][2] = 0
        self.save_lab_image(name, image_arr)
    
    def save_current_label(self, name_vis, name_label, data):
        #image_arr = np.copy(self.data) #此处的self.data是由self.open_image()得到的
        #plt.imshow(data)
        #plt.show()
        label_arr = np.copy(data)
        label_arr_vis = np.copy(data)
        #print('type of label_arr',type(label_arr))
        #print('shape of label_arr',label_arr.shape)
        for cluster in self.clusters:
            background_count = 0
            teeth_count = 0
            plaque_count = 0
            #求该类别中的每个像素点对应的label值(3类)，记录其类别的个数之后，执行少数服从多数
            for p in cluster.pixels: #用于画每个块的代码
                if(data[p[0]][p[1]] == 0):
                    background_count = background_count + 1
                elif(data[p[0]][p[1]] == 1):
                    teeth_count = teeth_count + 1
                else:
                    plaque_count = plaque_count + 1
            
            if(background_count >= teeth_count and background_count >= plaque_count):
                for p in cluster.pixels:
                    label_arr_vis[p[0]][p[1]] = 0
                    label_arr[p[0]][p[1]] = 0
            elif(teeth_count >= background_count and teeth_count >= plaque_count):
                for p in cluster.pixels:
                    label_arr_vis[p[0]][p[1]] = 125
                    label_arr[p[0]][p[1]] = 1
            else:
                for p in cluster.pixels:
                    label_arr_vis[p[0]][p[1]] = 255
                    label_arr[p[0]][p[1]] = 2
            label_arr_vis[cluster.h][cluster.w] = 200 #对于label_vis中的聚类中心，设为200
        '''
        plt.imshow(label_arr)
        plt.show()
        plt.imshow(label_arr_vis)
        plt.show()
        '''
        io.imsave(name_vis, label_arr_vis) 
        #用于可视化，判断结果的正确性，按照少数服从多数原则，用于保存0-125-255(700*700)，聚类格中心为0格式的label_vis
        io.imsave(name_label, label_arr)  
        #真正用来使用的数据，按照少数服从多数原则，用于保存0-1-2(700*700)格式的label

    def iterate_10times(self, name_string):
        self.init_clusters()
        self.move_clusters()
        
        root_path_str = './result_20200601/'
        for i in trange(10):
        #for i in trange(1):
            self.assignment()
            #print('self.clusters:',self.clusters)
            #print('type of self.clusters:',type(self.clusters))
            #print('lenth of self.clusters:',len(self.clusters))
            #print('shape of self.clusters:',shape(self.clusters))
            self.update_cluster()
            #name = 'lenna_M{m}_K{k}_loop{loop}.png'.format(loop=i, m=self.M, k=self.K)
            #name = './result_20190822/result1000/{name_string}_M{m}_K{k}_loop{loop}.png'.format(loop=i, name_string=name_string, m=self.M, k=self.K)
            name = root_path_str + 'result1000/{name_string}_M{m}_K{k}_loop{loop}.png'.format(loop=i, name_string=name_string, m=self.M, k=self.K)
            self.save_current_image(name)
            #用于保存原图像进行超像素分割之后的结果
            if(i==9):
                #name = './result_20190822/result1000_loop10/{name_string}_M{m}_K{k}_loop{loop}.png'.format(loop=i, name_string=name_string, m=self.M, k=self.K)
                name = root_path_str + 'result1000_loop10/{name_string}_M{m}_K{k}_loop{loop}.png'.format(loop=i, name_string=name_string, m=self.M, k=self.K)
                self.save_current_image(name)
            
            #labelname_vis = './result_20190822/label_vis1000/{name_string}_M{m}_K{k}_loop{loop}.png'.format(loop=i, name_string=name_string, m=self.M, k=self.K)
            #labelname = './result_20190822/label1000/{name_string}_M{m}_K{k}_loop{loop}.png'.format(loop=i, name_string=name_string, m=self.M, k=self.K)
            labelname_vis = root_path_str + 'label_vis1000/{name_string}_M{m}_K{k}_loop{loop}.png'.format(loop=i, name_string=name_string, m=self.M, k=self.K)
            labelname = root_path_str + 'label1000/{name_string}_M{m}_K{k}_loop{loop}.png'.format(loop=i, name_string=name_string, m=self.M, k=self.K)
            
            #labelpath = './labelnew/{name_string}.png'.format(loop=i, name_string=name_string)
            #labelpath = './SegmentationClass/{name_string}.png'.format(loop=i, name_string=name_string)
            labelpath = './for_super_pixels/SegmentationClass/{name_string}.png'.format(loop=i, name_string=name_string)
            #labelresult = self.open_image(labelpath)
            labelresult = io.imread(labelpath)
            #plt.imshow(labelresult)
            #plt.show()
            #print('type of labelresult',type(labelresult))
            #print('shape of labelresult',labelresult.shape)
            self.save_current_label(labelname_vis, labelname, labelresult)
            #用于保存groundtruth图像，按照原图像进行超像素分割之后对应的结果
            if(i==9):
                #labelname_vis = './result_20190822/label_vis1000_loop10/{name_string}_M{m}_K{k}_loop{loop}.png'.format(loop=i, name_string=name_string, m=self.M, k=self.K)
                #labelname = './result_20190822/label1000_loop10/{name_string}_M{m}_K{k}_loop{loop}.png'.format(loop=i, name_string=name_string, m=self.M, k=self.K)
                labelname_vis = root_path_str + 'label_vis1000_loop10/{name_string}_M{m}_K{k}_loop{loop}.png'.format(loop=i, name_string=name_string, m=self.M, k=self.K)
                labelname = root_path_str + 'label1000_loop10/{name_string}_M{m}_K{k}_loop{loop}.png'.format(loop=i, name_string=name_string, m=self.M, k=self.K)
                self.save_current_label(labelname_vis, labelname, labelresult)
            

    def save_mat_new(self, maxlen, name_string):
        root_path_str = './result_20200601/'

        #maxlen = 1024
        X = np.zeros((maxlen,1))
        Y = np.zeros((maxlen,1))
        Z = np.zeros((maxlen,1))
        R = np.zeros((maxlen,1))
        G = np.zeros((maxlen,1))
        B = np.zeros((maxlen,1))
        Point = []
        image_arr = np.copy(self.data)
        key_point_num = 0
        for cluster in self.clusters:
            #print(cluster.pixels)
            #print('type of cluster.pixels',type(cluster.pixels))
            for p in cluster.pixels:
                image_arr[p[0]][p[1]][0] = cluster.l
                image_arr[p[0]][p[1]][1] = cluster.a
                image_arr[p[0]][p[1]][2] = cluster.b
            Point.append(cluster.pixels)
            X[key_point_num] = cluster.h
            Y[key_point_num] = cluster.w
            #print('X[key_point_num]',X[key_point_num])
            #print('Y[key_point_num]',Y[key_point_num])
            key_point_num = key_point_num + 1
        rgb_arr = color.lab2rgb(image_arr)#image_arr是lab格式的图像；rgb_arr是rgb格式的图像
        '''
        rgb_arr的数据范围为[0-1]
        使用io.imsave()可以直接保存为[0-255]范围的图像
        '''
        #plt.imshow(rgb_arr)
        #plt.show()
        #print('type of rgb_arr',type(rgb_arr))
        #print('shape of rgb_arr', rgb_arr.shape)
        #print('key_point_num is', key_point_num)
        for i in range(key_point_num):
            inter_X = int(X[i][0])
            inter_Y = int(Y[i][0])
            '''
            X[i]获取到的是numpy类型的;
            X[i][0]获取到的是float类型，需要转化成int类型才能做为下标
            '''
            R[i] = rgb_arr[inter_X][inter_Y][0]
            G[i] = rgb_arr[inter_X][inter_Y][1]
            B[i] = rgb_arr[inter_X][inter_Y][2]
            Z[i] = 255 * (rgb_arr[inter_X][inter_Y][0] + rgb_arr[inter_X][inter_Y][1] + rgb_arr[inter_X][inter_Y][2])/3
            #Z[i]取的是该区域rgb值的均值，范围为[0-255]
        #result = {'X':X,'Y':Y,'Z':Z,'Point':Point}
        result = {'X':X,'Y':Y,'Z':Z,'R':R,'G':G,'B':B,'Point':Point}
        final_result = dict(result=result)

        #mat_name = './result_20190822/result_mat/{name_string}.mat'.format(name_string=name_string)
        mat_name = root_path_str + 'result_mat/{name_string}.mat'.format(name_string=name_string)
        scio.savemat(mat_name, {'shape':final_result}, appendmat=True, do_compression=True)

def pre_mkdir():
    #创建结果文件夹路径

    root_path_str = './result_20200601/'
    #result_mat_path = './result_20190822/result_mat/'
    result_mat_path = root_path_str + 'result_mat/'
    if(os.path.exists(result_mat_path) == False):
        os.mkdir(result_mat_path)
        print('result_mat_path is creating!')
    else:
        print('result_mat_path is created!')

    #result1000_path = './result_20190822/result1000/'
    result1000_path = root_path_str + 'result1000/'
    if(os.path.exists(result1000_path) == False):
        os.mkdir(result1000_path)
        print('result1000_path is creating!')
    else:
        print('result1000_path is created!')

    #result1000_loop10 = './result_20190822/result1000_loop10/'
    result1000_loop10 =  root_path_str + 'result1000_loop10/'
    if(os.path.exists(result1000_loop10) == False):
        os.mkdir(result1000_loop10)
        print('result1000_loop10 is creating!')
    else:
        print('result1000_loop10 is created!')
        
    #label1000_path = './result_20190822/label1000/'
    label1000_path =  root_path_str + 'label1000/'
    if(os.path.exists(label1000_path) == False):
        os.mkdir(label1000_path)
        print('label1000_path is creating!')
    else:
        print('label1000_path is created!')
        
    #label_vis1000_path = './result_20190822/label_vis1000/'
    label_vis1000_path =  root_path_str + 'label_vis1000/'
    if(os.path.exists(label_vis1000_path) == False):
        os.mkdir(label_vis1000_path)
        print('label_vis1000_path is creating!')
    else:
        print('label_vis1000_path is created!')
        
    #label1000_loop10_path = './result_20190822/label1000_loop10/'
    label1000_loop10_path =  root_path_str + 'label1000_loop10/'
    if(os.path.exists(label1000_loop10_path) == False):
        os.mkdir(label1000_loop10_path)
        print('label1000_loop10_path is creating!')
    else:
        print('label1000_loop10_path is created!')
        
    #label_vis1000_loop10_path = './result_20190822/label_vis1000_loop10/'
    label_vis1000_loop10_path =  root_path_str + 'label_vis1000_loop10/'
    if(os.path.exists(label_vis1000_loop10_path) == False):
        os.mkdir(label_vis1000_loop10_path)
        print('label_vis1000_loop10_path is creating!')
    else:
        print('label_vis1000_loop10_path is created!')   



if __name__ == '__main__':
    
    #root_path = os.path.abspath('./yuan/') #原图像所在路径
    root_path = os.path.abspath('./for_super_pixels/JPEGImages/') #原图像所在路径
    pre_mkdir()

    imglist = os.listdir(root_path) #获取文件夹中的全部文件序列
    imglist.sort() #对获取到文件夹中的文件进行排序

    print('length of imglist is', len(imglist))

    #for i in range(0, len(imglist)):
    #for i in range(20, len(imglist)):
    #for i in range(27, len(imglist)):
    #for i in range(147, len(imglist)):
    #for i in range(518, len(imglist)):
    #for i in range(637, len(imglist)):
    for i in range(1281, len(imglist)):
    #for i in range(605, len(imglist)):
    #for i in range(0, 1):
        image_name = os.path.join(root_path, imglist[i]) #原图像路径
        name_string = os.path.splitext(imglist[i])[0] #原图像名称前缀
        print('image_name is：', image_name)
        print('name_string is ', name_string)

        p = SLICProcessor(image_name, 1000, 5)
        p.iterate_10times(name_string)
        
        maxlen = 1024
        p.save_mat_new(maxlen, name_string)




#conda activate slic
