import datetime
import argparse

import torch
import torch.nn as nn
import numpy as np
from easydict import EasyDict

from model.network import TOTAL3D
from model.loss import ReconLoss, JointLoss, Detection_Loss, PoseLoss

from configs.data_config import Config as Data_Config
from configs.data_config import NYU37_TO_PIX3D_CLS_MAPPING, NYU40CLASSES
from model.utils.libs import get_rotation_matrix_gt, get_mask_status
from model.utils.libs import to_dict_tensor
from eval_metrics import *
from model.utils.libs import write_obj
from scipy.io import savemat
from model.utils.libs import *
import os


def parser():
    parser = argparse.ArgumentParser()

    # for mgn
    parser.add_argument("--bottleneck_size", type = int, default = 1024, help='dim_out_patch')
    parser.add_argument("--number_points", type = int, default = 2562)
    parser.add_argument("--subnetworks", type = int, default = 2, help='num of tnn subnetworks')
    parser.add_argument("--face_samples", type = int, default = 1, help='num of face_samples')
    parser.add_argument("--num_classes", type = int, default = 9, help='num of classes of dataset')
    parser.add_argument("--threshold", type = float, default = 0.001, help='threshold of tnn network')
    parser.add_argument("--factor", type = float, default = 0.5, help='factor of tnn network')

    parser.add_argument("--lr", type = float, default = 1e-3)
    parser.add_argument("--betas", type = list, default = [0.9, 0.999])
    parser.add_argument("--eps", type = float, default = 1e-08)
    parser.add_argument("--weight_decay", type = float, default = 1e-04)
    parser.add_argument("--batch_size", type = int, default = 32, help = 'Batch Size' )
    parser.add_argument("--nepoch", type = float, default = 500, help = 'the total training epochs')

    parser.add_argument("--mgn_load_path", type = str, default = "out", help = 'path of saved model')
    parser.add_argument("--len_load_path", type = str, default = "out", help = 'path of saved model')
    parser.add_argument("--odn_load_path", type = str, default = "out", help = 'path of saved model')

    parser.add_argument("--log_path", type = str, default = "log", help = 'path of log info')
    parser.add_argument("--name", type = str, default = "test_code", help = 'name of this training process')
    
    parser.add_argument("--demo", action="store_true", default = False, help = 'demo or not')
    parser.add_argument("--demo_path", type = str, default = 'demo')
    parser.add_argument("--check_freq", type = int, default = 5, help = 'The frequency of print loss in screen.')
    parser.add_argument("--save_freq", type = int, default = 10, help = 'The frequency of saving a model.')
    opt = parser.parse_args()
    opt = EasyDict(opt.__dict__)

    opt.date = str(datetime.datetime.now())
    return opt

class Trainer():
    def __init__(self, opt, device = None):
        self.opt = opt
        self.device = device
        self.model = TOTAL3D(opt)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = opt.lr, betas = opt.betas, eps = opt.eps, weight_decay=opt.weight_decay)

        self.Recon_Loss = ReconLoss()
        self.detloss = Detection_Loss()
        self.poseloss = PoseLoss()
        self.jointloss = JointLoss()

        dataset_config = Data_Config('sunrgbd')
        if device != torch.device("cpu"):
            if_cuda = True
        else:
            if_cuda = False
        self.bins_tensor = to_dict_tensor(dataset_config.bins, if_cuda=if_cuda)
        self.load_model()
    
    def load_model(self):
        len_load_path = self.opt.len_load_path
        odn_load_path = self.opt.odn_load_path
        mgn_load_path = self.opt.mgn_load_path
        t3d_load_path = self.opt.t3d_load_path
        if t3d_load_path != '':
            self.model.load_state_dict(torch.load(t3d_load_path, map_location=self.device))
            print("Loading Total3D model " + t3d_load_path)
        else:
            if len_load_path != '':
                self.model.len.load_state_dict(torch.load(len_load_path, map_location=self.device))
                print("Loading LEN model " + len_load_path)
            if odn_load_path != '':
                self.model.odn.load_state_dict(torch.load(odn_load_path, map_location=self.device))
                print("Loading ODN model " + odn_load_path)
            if mgn_load_path != '':
                self.model.mgn.load_state_dict(torch.load(mgn_load_path, map_location=self.device))
                print("Loading MGN model " + mgn_load_path)
        return

    def save_net(self, save_dir):
        torch.save(self.model.state_dict(), save_dir)
        return
 
    def train_step(self, data, train=True):
        len_input, odn_input, joint_input = self.to_device(data)
        len_est_data, odn_est_data, mgn_est_data = self.model(len_input, odn_input, joint_input, train)

        joint_input = dict(dict(len_input, **odn_input), **joint_input)
        # joint_est_data = dict(len_est_data.items() + odn_est_data.items() + mgn_est_data.items())
        joint_est_data = dict(dict(len_est_data, **odn_est_data), **mgn_est_data)
        if train == False:
            return joint_est_data, joint_input
        len_loss, layout_results = self.poseloss(len_est_data, len_input, self.bins_tensor)

        odn_loss = self.detloss(odn_est_data, odn_input)

        joint_loss, extra_results = self.jointloss(joint_est_data, joint_input, self.bins_tensor, layout_results)
        recon_loss = self.Recon_Loss(mgn_est_data, joint_input, extra_results)

        loss = odn_loss['total'] + recon_loss['mesh_loss'] + joint_loss['total_loss'] + len_loss['total']
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'total':loss, 
                'len loss': len_loss['total'], 
                'odn loss': odn_loss['total'], 
                'mgn loss': recon_loss['mesh_loss'], 
                'joint loss': joint_loss['total_loss']}
    
    def eval(self, data):
        len_input, odn_input, joint_input = self.to_device(data)
        len_est_data, odn_est_data, mgn_est_data = self.model(len_input, odn_input, joint_input, train=False, eval=True)

        joint_input = dict(dict(len_input, **odn_input), **joint_input)
        # joint_est_data = dict(len_est_data.items() + odn_est_data.items() + mgn_est_data.items())
        joint_est_data = dict(dict(len_est_data, **odn_est_data), **mgn_est_data)

        len_loss, layout_results = self.poseloss(len_est_data, len_input, self.bins_tensor)
        
        joint_est_data['lo_bdb3D'] = layout_results['lo_bdb3D_result']
        loss = self.get_metric_values(joint_est_data, joint_input)
        return loss


    def to_device(self, data):
        device = self.device

        image = data['image'].to(device)
        pitch_reg = data['camera']['pitch_reg'].float().to(device)
        pitch_cls = data['camera']['pitch_cls'].long().to(device)
        roll_reg = data['camera']['roll_reg'].float().to(device)
        roll_cls = data['camera']['roll_cls'].long().to(device)
        lo_ori_reg = data['layout']['ori_reg'].float().to(device)
        lo_ori_cls = data['layout']['ori_cls'].long().to(device)
        lo_centroid = data['layout']['centroid_reg'].float().to(device)
        lo_coeffs = data['layout']['coeffs_reg'].float().to(device)
        lo_bdb3D = data['layout']['bdb3D'].float().to(device)

        layout_input = {'image':image, 'pitch_reg':pitch_reg, 'pitch_cls':pitch_cls, 'roll_reg':roll_reg,
                        'roll_cls':roll_cls, 'lo_ori_reg':lo_ori_reg, 'lo_ori_cls':lo_ori_cls, 'lo_centroid':lo_centroid,
                        'lo_coeffs':lo_coeffs, 'lo_bdb3D':lo_bdb3D}

        patch = data['boxes_batch']['patch'].to(device)
        g_features = data['boxes_batch']['g_feature'].float().to(device)
        size_reg = data['boxes_batch']['size_reg'].float().to(device)
        size_cls = data['boxes_batch']['size_cls'].float().to(device) # The reason to use long is that we will cat it on our embeddings. 
        ori_reg = data['boxes_batch']['ori_reg'].float().to(device)
        ori_cls = data['boxes_batch']['ori_cls'].long().to(device)
        centroid_reg = data['boxes_batch']['centroid_reg'].float().to(device)
        centroid_cls = data['boxes_batch']['centroid_cls'].long().to(device)
        offset_2D = data['boxes_batch']['delta_2D'].float().to(device)
        split = data['obj_split']
        rel_pair_count = torch.cat([torch.tensor([0]), torch.cumsum(
            torch.pow(data['obj_split'][:,1]- data['obj_split'][:,0],2),dim = 0)],dim = 0)

        object_input = {'patch': patch, 'g_features': g_features, 'size_reg': size_reg, 'size_cls': size_cls,
                        'ori_reg': ori_reg, 'ori_cls': ori_cls, 'centroid_reg': centroid_reg, 'centroid_cls': centroid_cls,
                        'offset_2D': offset_2D, 'split': split, 'rel_pair_counts': rel_pair_count}
        
        bdb3D = data['boxes_batch']['bdb3D'].float().to(device)
        K = data['camera']['K'].float().to(device)
        depth_maps = [depth.float().to(device) for depth in data['depth']]

        # ground-truth camera rotation
        cam_R_gt = get_rotation_matrix_gt(self.bins_tensor,
                                            pitch_cls, pitch_reg,
                                            roll_cls, roll_reg)

        obj_masks = data['boxes_batch']['mask']

        mask_status = get_mask_status(obj_masks, split)

        mask_flag = 1
        if 1 not in mask_status:
            mask_flag = 0

        # Notice: we should conclude the NYU37 classes into pix3d (9) classes before feeding into the network.
        cls_codes = torch.zeros([size_cls.size(0), 9]).to(device)
        cls_codes[range(size_cls.size(0)), [NYU37_TO_PIX3D_CLS_MAPPING[cls.item()] for cls in
                                            torch.argmax(size_cls, dim=1)]] = 1

        patch_for_mesh = patch[mask_status.nonzero()]
        cls_codes_for_mesh = cls_codes[mask_status.nonzero()]
        # patch_for_mesh = patch
        # cls_codes_for_mesh = cls_codes

        '''calculate loss from the interelationship between object and layout.'''
        bdb2D_from_3D_gt = data['boxes_batch']['bdb2D_from_3D'].float().to(device)
        bdb2D_pos = data['boxes_batch']['bdb2D_pos'].float().to(device)

        joint_input = {'bdb3D':bdb3D, 'K':K, 'depth_maps':depth_maps, 'cam_R_gt':cam_R_gt, 
                        'obj_masks':obj_masks, 'mask_status':mask_status, 'mask_flag':mask_flag, 'patch_for_mesh':patch_for_mesh,
                        'cls_codes_for_mesh':cls_codes_for_mesh, 'bdb2D_from_3D_gt':bdb2D_from_3D_gt, 'bdb2D_pos':bdb2D_pos}

        return layout_input, object_input, joint_input

    def get_metric_values(self, est_data, gt_data):
        ''' Performs a evaluation step.
        '''

        # Layout IoU
        lo_bdb3D_out, _ = get_layout_bdb_sunrgbd(self.bins_tensor, est_data['lo_ori_reg'],
                                              torch.argmax(est_data['lo_ori_cls'], 1), est_data['lo_centroid'],
                                              est_data['lo_coeffs'])

        layout_iou = []
        for index in range(len(lo_bdb3D_out)):
        # for index, sequence_id in enumerate(gt_data['sequence_id']):
            lo_iou = get_iou_cuboid(lo_bdb3D_out[index, :, :].cpu().numpy(), gt_data['lo_bdb3D'][index, :, :].cpu().numpy())
            layout_iou.append(lo_iou)

        # camera orientation for evaluation
        cam_R_out = get_rotation_matix_result(self.bins_tensor,
                                              torch.argmax(est_data['pitch_cls'], 1), est_data['pitch_reg'],
                                              torch.argmax(est_data['roll_cls'], 1), est_data['roll_reg'])

        # projected center
        P_result = torch.stack(((gt_data['bdb2D_pos'][:, 0] + gt_data['bdb2D_pos'][:, 2]) / 2 - (
                gt_data['bdb2D_pos'][:, 2] - gt_data['bdb2D_pos'][:, 0]) * est_data['offset_2D'][:, 0],
                                (gt_data['bdb2D_pos'][:, 1] + gt_data['bdb2D_pos'][:, 3]) / 2 - (
                                        gt_data['bdb2D_pos'][:, 3] - gt_data['bdb2D_pos'][:, 1]) * est_data['offset_2D'][:, 1]), 1)

        bdb3D_out_form_cpu, bdb3D_out = get_bdb_evaluation(self.bins_tensor, torch.argmax(est_data['ori_cls'], 1), est_data['ori_reg'],
                                                       torch.argmax(est_data['centroid_cls'], 1), est_data['centroid_reg'],
                                                       gt_data['size_cls'], est_data['size_reg'], P_result, gt_data['K'], cam_R_out, gt_data['split'], return_bdb=True)

        bdb2D_out = get_bdb_2d_result(bdb3D_out, cam_R_out, gt_data['K'], gt_data['split'])

        nyu40class_ids = []
        IoU3D = []
        IoU2D = []
        class_list = [NYU40CLASSES[int(item['classid'])] for item in bdb3D_out_form_cpu]
        for index, evaluate_bdb in enumerate(bdb3D_out_form_cpu):
            
            NYU40CLASS_ID = int(evaluate_bdb['classid'])
            iou_3D = get_iou_cuboid(get_corners_of_bb3d_no_index(evaluate_bdb['basis'], evaluate_bdb['coeffs'],
                                                                 evaluate_bdb['centroid']),
                                    gt_data['bdb3D'][index, :, :].cpu().numpy())

            box1 = bdb2D_out[index, :].cpu().numpy()
            box2 = gt_data['bdb2D_from_3D_gt'][index, :].cpu().numpy()

            box1 = {'u1': box1[0], 'v1': box1[1], 'u2': box1[2], 'v2': box1[3]}
            box2 = {'u1': box2[0], 'v1': box2[1], 'u2': box2[2], 'v2': box2[3]}

            iou_2D = get_iou(box1, box2)

            nyu40class_ids.append(NYU40CLASS_ID)
            IoU3D.append(iou_3D)
            IoU2D.append(iou_2D)

        # '''Save results'''
        # if self.cfg.config['log']['save_results']:

        #     save_path = self.cfg.config['log']['vis_path']

        #     for index, sequence_id in enumerate(gt_data['sequence_id']):
        #         save_path_per_img = os.path.join(save_path, str(sequence_id.item()))
        #         if not os.path.exists(save_path_per_img):
        #             os.mkdir(save_path_per_img)

        #         # save layout results
        #         savemat(os.path.join(save_path_per_img, 'layout.mat'),
        #                 mdict={'layout': lo_bdb3D_out[index, :, :].cpu().numpy()})

        #         # save bounding boxes and camera poses
        #         interval = gt_data['split'][index].cpu().tolist()
        #         current_cls = nyu40class_ids[interval[0]:interval[1]]

        #         savemat(os.path.join(save_path_per_img, 'bdb_3d.mat'),
        #                 mdict={'bdb': bdb3D_out_form_cpu[interval[0]:interval[1]], 'class_id': current_cls})
        #         savemat(os.path.join(save_path_per_img, 'r_ex.mat'),
        #                 mdict={'cam_R': cam_R_out[index, :, :].cpu().numpy()})

        #         current_faces = est_data['out_faces'][interval[0]:interval[1]].cpu().numpy()
        #         current_coordinates = est_data['meshes'].transpose(1,2)[interval[0]:interval[1]].cpu().numpy()

        #         for obj_id, obj_cls in enumerate(current_cls):
        #             file_path = os.path.join(save_path_per_img, '%s_%s.obj' % (obj_id, obj_cls))

        #             mesh_obj = {'v': current_coordinates[obj_id],
        #                         'f': current_faces[obj_id]}

        #             write_obj(file_path, mesh_obj)

        metrics = {}
        metrics['layout_iou'] = np.mean(layout_iou)
        metrics['iou_3d'] = IoU3D
        metrics['iou_2d'] = IoU2D
        metrics['class'] = class_list
        return metrics



    
    
    
    