from tester_pymesh import Tester

import time
import datetime
import argparse
import os

import torch
from torch.utils.data import DataLoader
from easydict import EasyDict
from dataset.SunrgbdDataloader import SunDataset, collate_fn
from configs.data_config import NYU40CLASSES
from detection import TwoDBB
from PIL import Image
import numpy as np

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
    parser.add_argument("--len_load_path", type = str, default = "") # "out/len_pretrain_model.pth", help = 'path of saved odn model')
    parser.add_argument("--odn_load_path", type = str, default = "") # "out/odn_pretrain_model.pth", help = 'path of saved len model')
    parser.add_argument("--t3d_load_path", type = str, default = "out/t3d_checkpoints_epoch66.pth") # "out/t3d_checkpoints_epoch150.pth"), help = 'path of saved t3d model')
    parser.add_argument("--model_path", type=str, default="out", help='dir to save checkpoints')

    parser.add_argument("--log_path", type = str, default = "log", help = 'path of log info')
    parser.add_argument("--name", type = str, default = "total3d", help = 'name of this training process')
    
    parser.add_argument("--demo", action="store_true", default = False, help = 'demo or not')
    parser.add_argument("--demo_path", type = str, default = 'demo')
    parser.add_argument("--check_freq", type = int, default = 5, help = 'The frequency of print loss in screen.')
    parser.add_argument("--save_freq", type = int, default = 10, help = 'The frequency of saving a model.')
    parser.add_argument("--cuda", type =str, default = "cuda:0", help = 'Which GPU to use for training.')
    parser.add_argument("--cuda_num", type =int, default = 0, help = 'Which GPU to use for training.')

    parser.add_argument("--mode", type =str, default = 'normal', choices = ['normal', 'replace', 'add', 'exchange'], help = 'mode to run the code')
    parser.add_argument("--src_class", type =str, default = 'table', help = 'the class we want to replace')
    parser.add_argument("--target_class", type =str, default = 'sofa', help = 'the class we want to replace with')
    parser.add_argument('--detection_path', type =str, default='detection-pretrain/sunrgbd_model_95000.npz')
    parser.add_argument('--img_path', type =str, default='demo/3/img.jpg')
    parser.add_argument('--json_path', type =str, default='demo/3/detections.json')
    parser.add_argument('--add_img', type =str, default='demo/1/img.jpg')
    parser.add_argument('--add_box', type =list, default=[38, 304, 245, 496])
    parser.add_argument('--k', type =str, default='default', choices=['default','ours','txt'])
    parser.add_argument('--k_path', type =str, default='K.txt')
    
    opt = parser.parse_args()
    opt = EasyDict(opt.__dict__)

    opt.date = str(datetime.datetime.now())
    return opt

def get_random_data_from_sunrgbd():
    dataset = SunDataset(root_path='.', device= opt.device, mode='test')
    Train_loader = DataLoader(dataset, batch_size= opt.batch_size, collate_fn=collate_fn, shuffle = False)
    for idx, gt_data in enumerate(Train_loader):
        if idx == 0:
            break
    return gt_data

if __name__ == "__main__":
    opt = parser()
    if torch.cuda.is_available():
        opt.device = torch.device(opt.cuda)
        torch.cuda.set_device(opt.cuda_num) 
    else:
        opt.device = torch.device("cpu")
    
    tester = Tester(opt, device=opt.device)
    tester.model.to(opt.device)
    tester.model.eval()

    if opt.k == 'default':
        K = torch.FloatTensor([[529.5,   0.,  365.],
                                [0.,   529.5,  265.], 
                                [0.,     0.,     1. ]]
                                )
    elif opt.k == 'ours':
        # K = torch.FloatTensor([[2961, 0, 1079], 
        #                         [0, 2962, 1933], 
        #                         [0, 0, 1]])
        # K = torch.FloatTensor([[3453.80723446789, 0, 1509.77575786894], 
        #                         [0, 3450.11438003616, 2042.44171449600], 
        #                         [0, 0, 1]])
        # K = torch.FloatTensor([[3346.90380668512, 0, 2035.77624651872], 
        #                         [0, 3352.41685350737, 1553.89162781762], 
        #                         [0, 0, 1]])
        K = torch.FloatTensor([[ 1.42278701e+03, -4.60370724e-01,  6.62377447e+02],
                                [ 0.00000000e+00,  1.42106838e+03,  3.03415123e+02],
                                [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    else:
        txt_path = opt.k_path
        K = torch.FloatTensor(np.loadtxt(txt_path))

    now_time = str(datetime.datetime.now().replace(microsecond=0)).replace(' ','_').replace(':','-')
    save_path = os.path.join(opt.demo_path, now_time)
    os.mkdir(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # gt_data = tester.read_from_img(K, save_path=save_path)
    # gt_data = get_random_data_from_sunrgbd(save_path)
    gt_data = tester.read_from_json(opt.img_path, opt.json_path, K, save_path)
    with torch.no_grad():
        est_data, data = tester.step(gt_data)

    lo_bdb3D_out, cam_R_out, bdb3D_out_form_cpu, bdb3D_out = tester.calculate(est_data, data)
    
    # save results
    print("Saving...")
    nyu40class_ids = [int(evaluate_bdb['classid']) for evaluate_bdb in bdb3D_out_form_cpu]
    
    # save bounding boxes and camera poses
    interval = data['split'][0].cpu().tolist()
    current_cls = nyu40class_ids[interval[0]:interval[1]+1]

    # # save meshes
    current_faces = est_data['out_faces'][interval[0]:interval[1]].cpu().numpy()
    current_coordinates = est_data['meshes'].transpose(1, 2)[interval[0]:interval[1]].cpu().numpy()

    img_path = os.path.join(save_path, 'exp_img.png')
    img = gt_data['origin_image'][0].to("cpu")
    import torchvision.transforms as transforms
    img = transforms.ToPILImage()(img)
    img.save(img_path)

    file_path = '%s/recon.ply' % (save_path)
    tester.save_mesh(current_coordinates, current_faces, bdb3D_out_form_cpu, current_cls, file_path)
        

    
