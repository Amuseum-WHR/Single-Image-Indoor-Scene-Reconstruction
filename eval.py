from trainer import Trainer

import time
import datetime
import argparse
import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from easydict import EasyDict
from dataset.SunrgbdDataloader import SunDataset, collate_fn
from configs.data_config import NYU40CLASSES
from torch.utils.tensorboard import SummaryWriter

from tqdm import *


def parser():
    parser = argparse.ArgumentParser()

    # for mgn
    parser.add_argument("--bottleneck_size", type = int, default = 1024, help='dim_out_patch')
    parser.add_argument("--number_points", type = int, default = 2562)
    parser.add_argument("--subnetworks", type = int, default = 2, help='num of tnn subnetworks')
    parser.add_argument("--face_samples", type = int, default = 1, help='num of face_samples')
    parser.add_argument("--num_classes", type = int, default = 9, help='num of classes of dataset')
    parser.add_argument("--threshold", type = float, default = 0.2, help='threshold of tnn network')
    parser.add_argument("--factor", type = float, default = 0.5, help='factor of tnn network')

    parser.add_argument("--lr", type = float, default = 1e-3)
    parser.add_argument("--betas", type = list, default = [0.9, 0.999])
    parser.add_argument("--eps", type = float, default = 1e-08)
    parser.add_argument("--weight_decay", type = float, default = 1e-04)
    parser.add_argument("--batch_size", type = int, default = 1, help = 'Batch Size' )
    parser.add_argument("--nepoch", type = float, default = 1, help = 'the total training epochs')

    parser.add_argument("--mgn_load_path", type = str, default = "", help = 'path of saved mgn model')
    parser.add_argument("--len_load_path", type = str, default = "", help = 'path of saved odn model')
    parser.add_argument("--odn_load_path", type = str, default = "", help = 'path of saved len model')
    parser.add_argument("--t3d_load_path", type = str, default = "../Total3D/out/t3d_checkpoints_epoch66.pth", help = 'path of saved t3d model')
    parser.add_argument("--model_path", type=str, default="out", help='dir to save checkpoints')

    parser.add_argument("--log_path", type = str, default = "log", help = 'path of log info')
    parser.add_argument("--name", type = str, default = "total3d", help = 'name of this training process')
    
    parser.add_argument("--save_freq", type = int, default = 10, help = 'The frequency of saving a model.')
    parser.add_argument("--cuda", type =str, default = "cuda:0", help = 'Which GPU to use for training.')
    parser.add_argument("--cuda_num", type =int, default = 0, help = 'Which GPU to use for training.')
    opt = parser.parse_args()
    opt = EasyDict(opt.__dict__)

    opt.date = str(datetime.datetime.now())
    return opt


if __name__ == "__main__":
    opt = parser()
    if torch.cuda.is_available():
        opt.device = torch.device(opt.cuda)
        torch.cuda.set_device(opt.cuda_num) 
    else:
        opt.device = torch.device("cpu")
    
    Writer = SummaryWriter()
    dataset = SunDataset(root_path='.', device= opt.device, mode='test')
    Train_loader = DataLoader(dataset, batch_size= opt.batch_size, collate_fn=collate_fn, shuffle = True)
    trainer = Trainer(opt, device=opt.device)
    trainer.model.to(opt.device)
    trainer.model.eval()
    epochs = opt.nepoch
    stat = dict()
    class_iou = dict()
    blank = dict()
    for i in NYU40CLASSES:
        stat[i] = 0
        class_iou[i] = 0
        blank[i] = 0

    with torch.no_grad():
        for epoch in range(1):
            loop = tqdm(enumerate(Train_loader), total=len(Train_loader))
            for idx, gt_data in loop:
                metrics = trainer.eval(gt_data)
                loop.set_description(f'Epoch [{epoch}/{epochs}]')
                # loop.set_postfix(loss = steploss['total'])
                
                layout_iou = metrics['layout_iou']
                IoU3D = metrics['iou_3d']
                # IoU2D = metrics['iou_2d']
                class_list = metrics['class']
                for i in range(len(class_list)):
                    stat[class_list[i]] += 1
                    class_iou[class_list[i]] += IoU3D[i]
                if idx % 20 == 1:
                    for item in class_iou:
                        if stat[item] != 0:
                            blank[item] = class_iou[item] / stat[item]
                    print(blank)
    
