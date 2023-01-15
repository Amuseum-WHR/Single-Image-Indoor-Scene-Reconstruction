import datetime
import argparse
import os
import math
import torch
import torch.nn as nn
import numpy as np
from easydict import EasyDict
from detection import TwoDBB
from PIL import Image

from model.network import TOTAL3D

from configs.data_config import Config as Data_Config
from configs.data_config import NYU37_TO_PIX3D_CLS_MAPPING, NYU40CLASSES, RECON_3D_CLS
from model.utils.libs import get_rotation_matrix_gt, get_mask_status
from model.utils.libs import to_dict_tensor
import torchvision.transforms as transforms

class Relation_Config(object):
    def __init__(self):
        self.d_g = 64
        self.d_k = 64
        self.Nr = 16

HEIGHT_PATCH = 256
WIDTH_PATCH = 256
trans = transforms.Compose([
    transforms.Resize((HEIGHT_PATCH, WIDTH_PATCH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

non_trans = transforms.Compose([
    transforms.Resize((HEIGHT_PATCH, WIDTH_PATCH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

rel_cfg = Relation_Config()
d_model = int(rel_cfg.d_g/4)

class_to_code = dict()
for i in range(len(NYU40CLASSES)):
    class_to_code[NYU40CLASSES[i]] = i

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

class Tester():
    def __init__(self, opt, device = None):
        self.opt = opt
        self.device = device
        self.model = TOTAL3D(opt)

        dataset_config = Data_Config('sunrgbd')
        if device != torch.device("cpu"):
            if_cuda = True
        else:
            if_cuda = False
        self.bins_tensor = to_dict_tensor(dataset_config.bins, if_cuda=if_cuda)
        self.mode = opt.mode
        # self.load_good()
        self.load_model()

    def load_good(self):
        path = 'out/pretrained_model.pth'
        state_dict = torch.load(path, map_location=self.device)
        odn_state = dict()
        len_state = dict()
        mgn_state = dict()
        for item in state_dict:
            for i in state_dict[item]:
                if i[0:6] == 'layout':
                    new_name = i[25:]
                    len_state[new_name] = state_dict[item][i]
                elif i[0:4] == 'mesh':
                    new_name = i[27:]
                    if new_name[0:8] == 'decoders' or new_name[0:5] == 'error':
                        new_name = 'decoder.' + new_name
                    mgn_state[new_name] = state_dict[item][i]
                elif i[0:6] == 'object':
                    new_name = i[17:]
                    if new_name[0:6] == 'resnet':
                        new_name = new_name[0:6] + new_name[13:]
                    elif new_name[0:6] == 'relnet':
                        if new_name[7:11] == 'fc_K':
                            new_name = new_name[0:7] + 'K' + new_name[11:]
                        if new_name[7:11] == 'fc_Q':
                            new_name = new_name[0:7] + 'Q' + new_name[11:]
                        if new_name[7:11] == 'fc_g':
                            new_name = new_name[0:7] + 'G' + new_name[11:]
                        if new_name[7:13] == 'conv_s':
                            new_name = new_name[0:7] + 'scale_layer' + new_name[13:]
                    elif new_name[0:11] == 'fc_centroid':
                        new_name = 'fc4' + new_name[11:]
                    elif new_name[0:3] == 'fc5':
                        new_name = 'fc3' + new_name[3:]
                    elif new_name[0:3] == 'fc3':
                        new_name = 'fc7' + new_name[3:]
                    elif new_name[0:3] == 'fc4':
                        new_name = 'fc8' + new_name[3:]
                    elif new_name[0:8] == 'fc_off_1':
                        new_name = 'fc1' + new_name[8:]
                    elif new_name[0:8] == 'fc_off_2':
                        new_name = 'fc2' + new_name[8:]
                    elif new_name[0:3] == 'fc1':
                        new_name = 'fc5' + new_name[3:]
                    elif new_name[0:3] == 'fc2':
                        new_name = 'fc6' + new_name[3:]
                    odn_state[new_name] = state_dict[item][i]
        self.model.len.load_state_dict(len_state)
        self.model.odn.load_state_dict(odn_state)
        self.model.mgn.load_state_dict(mgn_state)
    
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
    
    def step(self, data):
        
        if self.mode == 'replace':
            src_cls = class_to_code[self.opt.src_class]
            target_cls = class_to_code[self.opt.target_class]
            return self.replace_step(data, src_cls, target_cls)
        else:
            if self.mode == 'exchange':
                self.ch1 = self.opt.src_class
                self.ch2 = self.opt.target_class
            return self.normal_step(data)
    
    def replace_step(self, data, src_cls, target_cls):
        len_input, odn_input, joint_input = self.replace_to_device(data, src_cls, target_cls)
        len_est_data, odn_est_data, mgn_est_data = self.model(len_input, odn_input, joint_input, train=False)

        joint_input = dict(dict(len_input, **odn_input), **joint_input)
        # joint_est_data = dict(len_est_data.items() + odn_est_data.items() + mgn_est_data.items())
        joint_est_data = dict(dict(len_est_data, **odn_est_data), **mgn_est_data)
        return joint_est_data, joint_input
 
    def normal_step(self, data):
        len_input, odn_input, joint_input = self.to_device(data)
        len_est_data, odn_est_data, mgn_est_data = self.model(len_input, odn_input, joint_input, train=False)

        joint_input = dict(dict(len_input, **odn_input), **joint_input)
        # joint_est_data = dict(len_est_data.items() + odn_est_data.items() + mgn_est_data.items())
        joint_est_data = dict(dict(len_est_data, **odn_est_data), **mgn_est_data)
        return joint_est_data, joint_input

    def replace_to_device(self, data, source_cls, target_cls):
        device = self.device

        image = data['image'].to(device)
        layout_input = {'image':image}

        patch = data['boxes_batch']['patch'].to(device)
        g_features = data['boxes_batch']['g_feature'].float().to(device)

        size_cls = data['boxes_batch']['size_cls'].float().to(device) # The reason to use long is that we will cat it on our embeddings. 
        for idx, cls in enumerate(torch.argmax(size_cls, dim=1)):
            if cls.item() == source_cls:
                size_cls[idx][source_cls] = 0
                size_cls[idx][target_cls] = 1
        split = data['obj_split']
        rel_pair_count = torch.cat([torch.tensor([0]), torch.cumsum(
            torch.pow(data['obj_split'][:,1]- data['obj_split'][:,0],2),dim = 0)],dim = 0)

        object_input = {'patch': patch, 'g_features': g_features, 'size_cls': size_cls,
                        'split': split, 'rel_pair_counts': rel_pair_count}

        K = data['camera']['K'].float().to(device)

        # # Notice: we should conclude the NYU37 classes into pix3d (9) classes before feeding into the network.
        cls_codes = torch.zeros([size_cls.size(0), 9]).to(device)
        cls_codes[range(size_cls.size(0)), [NYU37_TO_PIX3D_CLS_MAPPING[cls.item()] for cls in
                                            torch.argmax(size_cls, dim=1)]] = 1

        
        bdb2D_pos = data['boxes_batch']['bdb2D_pos'].float().to(device)

        joint_input = {'patch_for_mesh':patch,
                        'cls_codes_for_mesh':cls_codes,
                        'bdb2D_pos':bdb2D_pos,
                        'K':K}

        return layout_input, object_input, joint_input

    def process_raw_data(self, data, add_img=None):
        camera = data['camera']
        camera['K'] = camera['K'].unsqueeze(0)
        image = data['image']
        origin_image = transforms.ToTensor()(data['image']).unsqueeze(0)
        boxes = dict()

        boxes['bdb2D_pos'] = np.zeros((len(data['boxes']),4))
        boxes['size_cls'] = []
        for i in range(len(data['boxes'])):
            boxes['bdb2D_pos'][i,:] = np.array(data['boxes'][i]['bbox'])
            if type(data['boxes'][i]['class'])==str:
                boxes['size_cls'].append(class_to_code[data['boxes'][i]['class']])
            else:
                boxes['size_cls'].append(data['boxes'][i]['class'])

        n_objects = boxes['bdb2D_pos'].shape[0]

        g_feature = [[((loc2[0] + loc2[2]) / 2. - (loc1[0] + loc1[2]) / 2.) / (loc1[2] - loc1[0]),
                      ((loc2[1] + loc2[3]) / 2. - (loc1[1] + loc1[3]) / 2.) / (loc1[3] - loc1[1]),
                      math.log((loc2[2] - loc2[0]) / (loc1[2] - loc1[0])),
                      math.log((loc2[3] - loc2[1]) / (loc1[3] - loc1[1]))] \
                     for id1, loc1 in enumerate(boxes['bdb2D_pos'])
                     for id2, loc2 in enumerate(boxes['bdb2D_pos'])]

        locs = [num for loc in g_feature for num in loc]

        pe = torch.zeros(len(locs), d_model)
        position = torch.from_numpy(np.array(locs)).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        boxes['g_feature'] = pe.view(n_objects * n_objects, rel_cfg.d_g)

        # encode class
        cls_codes = torch.zeros([len(boxes['size_cls']), len(NYU40CLASSES)])
        cls_codes[range(len(boxes['size_cls'])), boxes['size_cls']] = 1
        boxes['size_cls'] = cls_codes

        patch = []
        for bdb in boxes['bdb2D_pos']:
            img = image.crop((bdb[0], bdb[1], bdb[2], bdb[3]))
            img = trans(img)
            patch.append(img)

        if add_img != None:
            add_img = trans(add_img)
            patch[-1] = add_img

        interval_list = [len(patch)]
        obj_split = torch.tensor([[sum(interval_list[:i]), sum(interval_list[:i+1])] for i in range(len(interval_list))])
        boxes['patch'] = torch.stack(patch)
        boxes['bdb2D_pos'] = torch.FloatTensor(boxes['bdb2D_pos'])
        image = non_trans(image).unsqueeze(0)

        gt_data = {
            'image': image,
            'boxes_batch': boxes,
            'obj_split': obj_split,
            'camera':camera,
            'origin_image': origin_image
        }
        return gt_data


    def to_device(self, data):
        device = self.device

        image = data['image'].to(device)

        layout_input = {'image':image}

        patch = data['boxes_batch']['patch'].to(device)
        g_features = data['boxes_batch']['g_feature'].float().to(device)
        size_cls = data['boxes_batch']['size_cls'].float().to(device) # The reason to use long is that we will cat it on our embeddings. 
        split = data['obj_split']
        rel_pair_count = torch.cat([torch.tensor([0]), torch.cumsum(
            torch.pow(data['obj_split'][:,1]- data['obj_split'][:,0],2),dim = 0)],dim = 0)

        object_input = {'patch': patch, 'g_features': g_features, 'size_cls': size_cls,
                        'split': split, 'rel_pair_counts': rel_pair_count}

        K = data['camera']['K'].float().to(device)

        # # Notice: we should conclude the NYU37 classes into pix3d (9) classes before feeding into the network.
        cls_codes = torch.zeros([size_cls.size(0), 9]).to(device)
        cls_codes[range(size_cls.size(0)), [NYU37_TO_PIX3D_CLS_MAPPING[cls.item()] for cls in
                                            torch.argmax(size_cls, dim=1)]] = 1

        bdb2D_pos = data['boxes_batch']['bdb2D_pos'].float().to(device)

        joint_input = { 'patch_for_mesh':patch,
                        'cls_codes_for_mesh':cls_codes,
                        'bdb2D_pos':bdb2D_pos,
                        'K':K}

        return layout_input, object_input, joint_input
    
    def calculate(self, est_data, data):
        from model.utils.libs import get_layout_bdb_sunrgbd, get_rotation_matix_result, get_bdb_evaluation
        bins_tensor = self.bins_tensor


        # lo_bdb3D_out, lo_bdb3D_out_form_cpu \
        lo_bdb3D_out, _ = get_layout_bdb_sunrgbd(bins_tensor, est_data['lo_ori_reg'],
                                            torch.argmax(est_data['lo_ori_cls'], 1),
                                            est_data['lo_centroid'],
                                            est_data['lo_coeffs'])
        
        cam_R_out = get_rotation_matix_result(bins_tensor,
                                            torch.argmax(est_data['pitch_cls'], 1), est_data['pitch_reg'],
                                            torch.argmax(est_data['roll_cls'], 1), est_data['roll_reg'])
        
        P_result = torch.stack(((data['bdb2D_pos'][:, 0] + data['bdb2D_pos'][:, 2]) / 2 -
                                (data['bdb2D_pos'][:, 2] - data['bdb2D_pos'][:, 0]) * est_data['offset_2D'][:, 0],
                                (data['bdb2D_pos'][:, 1] + data['bdb2D_pos'][:, 3]) / 2 -
                                (data['bdb2D_pos'][:, 3] - data['bdb2D_pos'][:, 1]) * est_data['offset_2D'][:,1]), 1)
        
        bdb3D_out_form_cpu, bdb3D_out = get_bdb_evaluation(bins_tensor,
                                                        torch.argmax(est_data['ori_cls'], 1),
                                                        est_data['ori_reg'],
                                                        torch.argmax(est_data['centroid_cls'], 1),
                                                        est_data['centroid_reg'],
                                                        data['size_cls'], est_data['size_reg'], P_result,
                                                        data['K'], cam_R_out, data['split'], return_bdb=True)

        if self.mode == 'exchange':
            class_list = [NYU40CLASSES[int(item['classid'])] for item in bdb3D_out_form_cpu]
            idx1 = class_list.index(self.ch1) if self.ch1 in class_list else -1
            idx2 = class_list.index(self.ch2) if self.ch2 in class_list else -1
            if idx1 == -1:
                print('can not find {} in the image'.format(self.ch1))
            elif idx2 == -1:
                print('can not find {} in the image'.format(self.ch2))
            else:
                tmp = bdb3D_out_form_cpu[idx1]['centroid']
                bdb3D_out_form_cpu[idx1]['centroid'] = bdb3D_out_form_cpu[idx2]['centroid']
                bdb3D_out_form_cpu[idx2]['centroid'] = tmp
                print('exchange the position of {} and {} in the image'.format(self.ch1, self.ch2))

        return lo_bdb3D_out, cam_R_out, bdb3D_out_form_cpu, bdb3D_out
    
    def get_data_from_image(self, gpu, image_path, model_path, K, add_box=None, save_path=None):
        raw_data = dict()
        raw_data['camera'] = {'K': K}
        raw_data['image'] = Image.open(image_path ,mode='r')
        print("2D detecting......")
        raw_data['boxes'] = TwoDBB(gpu, model_path, image_path)
        
        if add_box != None:
            raw_data['boxes'] = raw_data['boxes']  + add_box

        import json
        boxes = raw_data['boxes']
        for i in range(len(boxes)):
            for j in range(len(boxes[i]['bbox'])):
                boxes[i]['bbox'][j] = float(boxes[i]['bbox'][j])
            boxes[i]['class'] = float(boxes[i]['class'])
        print(save_path)
        with open(os.path.join(save_path, 'box.json'), 'w') as f:
            f.write(json.dumps(boxes))
        return raw_data
    
    def read_from_img(self, K, save_path=None):
        if self.mode != 'add':
            raw_data = self.get_data_from_image(self.opt.cuda_num, self.opt.img_path, self.opt.detection_path, K, save_path=save_path)
            gt_data = self.process_raw_data(raw_data)
        else:
            add_box = TwoDBB(self.opt.cuda_num, self.opt.detection_path, self.opt.add_img)
            zeor_cls = [0,1,2,11,12,18,19,20,21,22,23,28,31,33,34,35,36]
            idx = 0
            for i in range(len(add_box)):
                if add_box[i]['class'] not in zeor_cls:
                    idx = i
                    break
            print(idx)
            print("adding one {} in {} into {}...".format(NYU40CLASSES[add_box[idx]['class']], self.opt.add_img, self.opt.img_path))
            add_img = Image.open(self.opt.add_img ,mode='r')
            add_img = add_img.crop((add_box[idx]['bbox'][0], add_box[idx]['bbox'][1], add_box[idx]['bbox'][2], add_box[idx]['bbox'][3]))
            add_box = add_box[idx:idx+1]
            add_box[0]['bbox'] = self.opt.add_box
            raw_data = self.get_data_from_image(self.opt.cuda_num, self.opt.img_path, self.opt.detection_path, K, add_box, save_path=save_path)
            gt_data = self.process_raw_data(raw_data, add_img)
        return gt_data

    def read_from_json(self, image_path, json_path, K, save_path=None):
        if self.mode != 'add':
            raw_data = self.get_data_from_json(image_path, json_path, K, save_path = save_path)
            gt_data = self.process_raw_data(raw_data)
        else:
            add_box = TwoDBB(self.opt.cuda_num, self.opt.detection_path, self.opt.add_img)
            zeor_cls = [0,1,2,11,12,18,19,20,21,22,23,28,31,33,34,35,36]
            idx = 0
            for i in range(len(add_box)):
                if add_box[i]['class'] not in zeor_cls:
                    idx = i
                    break
            print(idx)
            print("adding one {} in {} into {}...".format(NYU40CLASSES[add_box[idx]['class']], self.opt.add_img, image_path))
            add_img = Image.open(self.opt.add_img ,mode='r')
            add_img = add_img.crop((add_box[idx]['bbox'][0], add_box[idx]['bbox'][1], add_box[idx]['bbox'][2], add_box[idx]['bbox'][3]))
            add_box = add_box[idx:idx+1]
            add_box[0]['bbox'] = self.opt.add_box
            raw_data = self.get_data_from_json(image_path, json_path, K, add_box, save_path)
            gt_data = self.process_raw_data(raw_data, add_img)
        return gt_data

    def get_data_from_json(self, image_path, json_path, K, add_box=None, save_path=None):
        raw_data = dict()
        raw_data['camera'] = {'K': K}
        raw_data['image'] = Image.open(image_path, mode='r')
        print("2D detecting......")
        import json
        raw_data['boxes'] = json.load(open(json_path))
        for i in range(len(raw_data['boxes'])):
            raw_data['boxes'][i]['class'] = class_to_code[raw_data['boxes'][i]['class']]

        if add_box != None:
            raw_data['boxes'] = raw_data['boxes'] + add_box

        boxes = raw_data['boxes']
        for i in range(len(boxes)):
            for j in range(len(boxes[i]['bbox'])):
                boxes[i]['bbox'][j] = float(boxes[i]['bbox'][j])
            boxes[i]['class'] = float(boxes[i]['class'])
        with open(os.path.join(save_path, 'box.json'), 'w') as f:
            f.write(json.dumps(boxes))
        return raw_data
