from __future__ import print_function

import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import cv2

from torchvision import transforms
from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.score import SegmentationMetric
from core.utils.visualize import get_color_pallete
from core.utils.logger import setup_logger
from core.utils.distributed import synchronize, get_rank, make_data_sampler, make_batch_data_sampler

from train import parse_args
import pandas as pd


import logging
logger = logging.getLogger("fatherModule")
logger.setLevel(level = logging.INFO)


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.metric = SegmentationMetric(3)

    def eval(self):
        self.metric.reset()
        image_pre_path = '../datasets/evaluation/result_image_20200728_deep_512_hks_RGB_XY_100_166_win/'
        image_gt_path = '../datasets/teeth/teeth0602/SegmentationClass/'
        
        image_pre_list = os.listdir(image_pre_path)
        name_list = []
        mIOU_list = []
        acc_list = []

        all_pixAcc = 0
        all_mIoU = 0
        all_IoU_0 = 0
        all_IoU_1 = 0
        all_IoU_2 = 0
        i = 0
        for image_pre_i in image_pre_list:
            print('image_pre_i:', image_pre_i)
            name_list.append(image_pre_i[:-4])
            print(image_pre_path+image_pre_i)
            image_pre = cv2.imread(image_pre_path+image_pre_i, cv2.IMREAD_GRAYSCALE)
            print(image_pre)
            print(type(image_pre))
            print(image_pre.shape)

            print(image_gt_path+image_pre_i)
            target = cv2.imread(image_gt_path+image_pre_i, cv2.IMREAD_GRAYSCALE)
            print(target)
            print(type(target))
            print(target.shape)
            # print('image_pre[0]: ', image_pre[0])
            # print('target[0]: ', target[0])
            image_pre_t = torch.Tensor(image_pre)
            target_t = torch.Tensor(target)

            # self.metric.update(list(image_pre), list(target))
            # self.metric.update(torch.from_numpy(image_pre), torch.from_numpy(target))
            self.metric.update(image_pre_t, target_t)
            #pixAcc, mIoU = self.metric.get()
            pixAcc, mIoU, IoU_0, IoU_1, IoU_2 = self.metric.get()
            #logger.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
            #    i + 1, pixAcc * 100, mIoU * 100))
            logger.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}, IoU_0: {:.3f}, IoU_1: {:.3f}, IoU_2: {:.3f}".format(
                i + 1, pixAcc * 100, mIoU * 100, IoU_0 * 100, IoU_1 * 100, IoU_2 * 100))
            all_pixAcc = all_pixAcc + pixAcc
            all_mIoU = all_mIoU + mIoU
            all_IoU_0 = all_IoU_0 + IoU_0
            all_IoU_1 = all_IoU_1 + IoU_1
            all_IoU_2 = all_IoU_2 + IoU_2
            mIOU_list.append(mIoU)
            acc_list.append(pixAcc)
            i += 1
            # if(i>10):
            #     break

        print('mean pixAcc: ', all_pixAcc / len(image_pre_list))
        print('mean mIoU: ', all_mIoU / len(image_pre_list))
        print('mean IoU_0: ', all_IoU_0 / len(image_pre_list))
        print('mean IoU_1: ', all_IoU_1 / len(image_pre_list))
        print('mean IoU_2: ', all_IoU_2 / len(image_pre_list))
        print('name_list: ', name_list)
        print('mIOU_list: ', mIOU_list)
        print('acc_list: ', acc_list)
        df_data = name_list + mIOU_list + acc_list
        title_name = ['image_name']
        df = pd.DataFrame(columns=title_name, data=df_data)
        df.to_csv('name.csv')
        
        # synchronize()


if __name__ == '__main__':
    args = parse_args()
    
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    '''

    # TODO: optim code
    args.save_pred = True
    if args.save_pred:
        outdir = '../runs/pred_pic/{}_{}_{}'.format(args.model, args.backbone, args.dataset)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(),
                          filename='{}_{}_{}_log.txt'.format(args.model, args.backbone, args.dataset), mode='a+')
    '''
    evaluator = Evaluator(args)
    evaluator.eval()
    torch.cuda.empty_cache()
