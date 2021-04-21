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

from torchvision import transforms
from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.score import SegmentationMetric
from core.utils.visualize import get_color_pallete
from core.utils.logger import setup_logger
from core.utils.distributed import synchronize, get_rank, make_data_sampler, make_batch_data_sampler

from train import parse_args
import scipy.io as scio #pang_add

class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        # dataset and dataloader
        #val_dataset = get_segmentation_dataset(args.dataset, split='val_train', mode='testval', transform=input_transform)
        #val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='testval', transform=input_transform)
        #val_dataset = get_segmentation_dataset(args.dataset, split='Aval', mode='testval', transform=input_transform)
        #val_dataset = get_segmentation_dataset(args.dataset, split='Bval', mode='testval', transform=input_transform)
        #val_dataset = get_segmentation_dataset(args.dataset, split='Cval', mode='testval', transform=input_transform)
        # val_dataset = get_segmentation_dataset(args.dataset, split='Dval', mode='testval', transform=input_transform)
        val_dataset = get_segmentation_dataset(args.dataset, split='val_0813', mode='testval', transform=input_transform)
        #val_dataset = get_segmentation_dataset(args.dataset, split='val_test', mode='testval', transform=input_transform)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,
                                            aux=args.aux, pretrained=True, pretrained_base=False,
                                            local_rank=args.local_rank,
                                            norm_layer=BatchNorm2d).to(self.device)
        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank], output_device=args.local_rank)
        self.model.to(self.device)

        self.metric = SegmentationMetric(val_dataset.num_class)

    def eval(self):
        self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        all_pixAcc = 0
        all_mIoU = 0
        all_IoU_0 = 0
        all_IoU_1 = 0
        all_IoU_2 = 0

        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                #outputs = model(image)
                outputs, mat_result_512, mat_result_3 = model(image)
            
            mat_result_512 = mat_result_512.transpose((1,2,0))
            mat_result_3 = mat_result_3.transpose((1,2,0))
            print('mat_result_512.type:',type(mat_result_512))
            print('mat_result_512.shape:', mat_result_512.shape)
            print('mat_result_3.type:',type(mat_result_3))
            print('mat_result_3.shape:', mat_result_3.shape)
            
            filename_mat = os.path.splitext(filename[0])[0] + '.mat'

            print('This is:', filename_mat)
            datapath_512 = '/home/pzn/pzncode/non-local/awesome-semantic-segmentation-pytorch/runs/mat_result_512_121/' + filename_mat
            print(datapath_512)
            result_512 = {'pzn_feature_512':mat_result_512}
            scio.savemat(datapath_512, {'feature512':result_512}, appendmat=True,  do_compression=True) 

            print('This is:', filename_mat)
            datapath_3 = '/home/pzn/pzncode/non-local/awesome-semantic-segmentation-pytorch/runs/mat_result_3_121/' + filename_mat
            print(datapath_3)
            result_3 = {'pzn_feature_3':mat_result_3}
            scio.savemat(datapath_3, {'feature3':result_3}, appendmat=True,  do_compression=True)


            self.metric.update(outputs[0], target)
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

            if self.args.save_pred:
                pred = torch.argmax(outputs[0], 1)
                pred = pred.cpu().data.numpy()

                predict = pred.squeeze(0)
                mask = get_color_pallete(predict, self.args.dataset)
                mask.save(os.path.join(outdir, os.path.splitext(filename[0])[0] + '.png'))
        print('mean pixAcc: ', all_pixAcc / len(self.val_loader))
        print('mean mIoU: ', all_mIoU / len(self.val_loader))
        print('mean IoU_0: ', all_IoU_0 / len(self.val_loader))
        print('mean IoU_1: ', all_IoU_1 / len(self.val_loader))
        print('mean IoU_2: ', all_IoU_2 / len(self.val_loader))
        synchronize()


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

    # TODO: optim code
    args.save_pred = True
    if args.save_pred:
        outdir = '../runs/pred_pic/{}_{}_{}'.format(args.model, args.backbone, args.dataset)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(),
                          filename='{}_{}_{}_log.txt'.format(args.model, args.backbone, args.dataset), mode='a+')

    evaluator = Evaluator(args)
    evaluator.eval()
    torch.cuda.empty_cache()
