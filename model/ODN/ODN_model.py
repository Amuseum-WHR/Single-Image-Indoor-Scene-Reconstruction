'''
Objection Detection Net
Based on Total3DUnderstanding by ynie
'''

import torch
import torch.nn as nn
import model.utils.resnet as resnet
import torch.utils.model_zoo as model_zoo
from model.ODN.Relation_Net import RelationNet
from configs.data_config import NYU40CLASSES

# class ODN(nn.Module):
#     def __init__(self, cfg):
#         super(ODN, self).__init__()
#         bin = cfg.bins
#         self.ORI_BIN_NUM = len(bin['ori_bin'])
#         self.CENTER_BIN_NUM = len(bin['centroid_bin'])

#         self.resnet = resnet.resnet34_Half(pretrained = False)

#         # set up relational network blocks
#         self.relnet = RelationNet()

#         # branch to predict the size
#         self.fc1 = nn.Linear(2048 + len(NYU40CLASSES), 128)
#         self.fc2 = nn.Linear(128, 3)

#         # branch to predict the orientation
#         self.fc3 = nn.Linear(2048 + len(NYU40CLASSES), 128)
#         self.fc4 = nn.Linear(128, self.ORI_BIN_NUM * 2)

#         # branch to predict the centroid
#         self.fc5 = nn.Linear(2048 + len(NYU40CLASSES), 128)
#         self.fc_centroid = nn.Linear(128, self.CENTER_BIN_NUM * 2)

#         # branch to predict the 2D offset
#         self.fc_off_1 = nn.Linear(2048 + len(NYU40CLASSES), 128)
#         self.fc_off_2 = nn.Linear(128, 2)

#         self.relu_1 = nn.LeakyReLU(0.2)
#         self.dropout_1 = nn.Dropout(p=0.5)
        
#     def forward(self, x, g_features, split, rel_pair_counts, size_cls):
#         '''
#             x: Patch x Channel x Hight x Width
#             Geometry_feature: sum of pairs x 64
#             split: e.g. [[0,5],[5,8]]
#             pair_counts: e.g. [0,49,113]
#             target: Patch x num of class: target[i]: object i should belongs to class[i], where targets[i][class[i]] =1 (?)
#         '''
#         a_features = self.resnet(x)
#         a_features = a_features.view(a_features.size(0), -1)

#         # extract relational features from other objects.
#         r_features = self.relnet(a_features, g_features, split, rel_pair_counts)

#         a_r_features = torch.add(a_features, r_features)

#         # add object category information
#         a_r_features = torch.cat([a_r_features, size_cls], 1)

#         # branch to predict the size
#         size = self.fc1(a_r_features)
#         size = self.relu_1(size)
#         size = self.dropout_1(size)
#         size = self.fc2(size)

#         # branch to predict the orientation
#         ori = self.fc3(a_r_features)
#         ori = self.relu_1(ori)
#         ori = self.dropout_1(ori)
#         ori = self.fc4(ori)
#         ori = ori.view(-1, self.ORI_BIN_NUM, 2)
#         ori_reg = ori[:, :, 0]
#         ori_cls = ori[:, :, 1]

#         # branch to predict the centroid
#         centroid = self.fc5(a_r_features)
#         centroid = self.relu_1(centroid)
#         centroid = self.dropout_1(centroid)
#         centroid = self.fc_centroid(centroid)
#         centroid = centroid.view(-1, self.CENTER_BIN_NUM, 2)
#         centroid_cls = centroid[:, :, 0]
#         centroid_reg = centroid[:, :, 1]

#         # branch to predict the 2D offset
#         offset = self.fc_off_1(a_r_features)
#         offset = self.relu_1(offset)
#         offset = self.dropout_1(offset)
#         offset = self.fc_off_2(offset)

#         object_output = {'size_reg':size, 'ori_reg':ori_reg,
#                              'ori_cls':ori_cls, 'centroid_reg':centroid_reg,
#                              'centroid_cls':centroid_cls, 'offset_2D':offset}
#         return object_output

# class ODN(nn.Module):
#     def __init__(self, cfg):
#         super(ODN, self).__init__()
#         bin = cfg.bins
#         self.ORI_BIN_NUM = len(bin['ori_bin'])
#         self.CENTER_BIN_NUM = len(bin['centroid_bin'])

#         self.resnet = resnet.resnet34_Half(pretrained = False)
#         # set up relational network blocks
#         self.relnet = RelationNet()
#         Linear_input_size = 2048 + len(NYU40CLASSES) # Depend on the dataset

#         # For Projection offset (2D)
#         self.fc1 = nn.Linear(Linear_input_size, 128)
#         # self.fc2 = nn.Linear(128, 3)
#         self.fc2 = nn.Linear(128, 2)
        
#         # For distance to Camara Center, which is used to calculate Centroid of 3D Box 
#         self.fc3 = nn.Linear(Linear_input_size, 128)
#         self.fc4 = nn.Linear(128, 2 * self.CENTER_BIN_NUM)

#         # For the size of 3D Box
#         self.fc5 = nn.Linear(Linear_input_size, 128)
#         self.fc6 = nn.Linear(128, 3)
#         # self.fc6 = nn.Linear(128, self.OBJ_CENTER_BIN * 2)

#         # For orientation of 3D box (1 dimension)
#         self.fc7 = nn.Linear(Linear_input_size, 128)
#         self.fc8 = nn.Linear(128, 2 * self.ORI_BIN_NUM)

#         self.relu = nn.LeakyReLU(0.2)
#         # self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p = 0.5)
#         # # Load Pretrained ResNet Model
#         pretrained_dict = model_zoo.load_url(resnet.model_urls['resnet34'])
#         model_dict = self.resnet.state_dict()
#         pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
#         model_dict.update(pretrained_dict)
#         self.resnet.load_state_dict(model_dict)

#         # initialize weights (I don't understand why it is necessary)
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0, 0.01)
#                 if hasattr(m.bias, 'data'):
#                     m.bias.data.zero_()

#     def forward(self, x, g_features, split, rel_pair_counts, size_cls):
#         '''
#             x: Patch x Channel x Hight x Width
#             Geometry_feature: sum of pairs x 64
#             split: e.g. [[0,5],[5,8]]
#             pair_counts: e.g. [0,49,113]
#             target: Patch x num of class: target[i]: object i should belongs to class[i], where targets[i][class[i]] =1 (?)
#         '''
#         # a_features = self.resnet(x)
#         # a_features = a_features.view(a_features.size(0), -1)

#         # # extract relational features from other objects.
#         # r_features = self.relnet(a_features, g_features, split, rel_pair_counts)

#         # a_r_features = torch.add(a_features, r_features)

#         # # add object category information
#         # a_r_features = torch.cat([a_r_features, size_cls], 1)

#         a_featrues = self.resnet(x)
#         a_features = a_featrues.view(a_featrues.size(0),-1) # N x 2048
#         r_features = self.relnet(a_featrues, g_features, split, rel_pair_counts) # N x 2048
#         a_r_features = torch.add(a_features, r_features) # N x 2048
#         a_r_features = torch.cat([a_r_features, size_cls], dim = 1) # N x (2048 + class)

#         # branch to predict the 2D offset
#         # offset = self.fc1(a_r_features)
#         # offset = self.relu(offset)
#         # offset = self.dropout(offset)
#         # offset = self.fc2(offset)

#         offset = self.fc1(a_r_features)
#         offset = self.relu(offset)
#         offset = self.dropout(offset)
#         offset = self.fc2(offset)

#         # branch to predict the centroid
#         # centroid = self.fc3(a_r_features)
#         # centroid = self.relu(centroid)
#         # centroid = self.dropout(centroid)
#         # centroid = self.fc4(centroid)
#         # centroid = centroid.view(-1, self.CENTER_BIN_NUM, 2)
#         # centroid_cls = centroid[:, :, 0]
#         # centroid_reg = centroid[:, :, 1]

#         distance = self.fc3(a_r_features)
#         distance = self.relu(distance)
#         distance = self.dropout(distance)
#         distance = self.fc4(distance)
#         distance = distance.view(-1, self.CENTER_BIN_NUM, 2)
#         distance_cls = distance[:,:,0]
#         distance_reg = distance[:,:,1]

#         # branch to predict the size
#         size = self.fc5(a_r_features)
#         size = self.relu(size)
#         size = self.dropout(size)
#         size = self.fc6(size)

#         # branch to predict the orientation
#         # ori = self.fc7(a_r_features)
#         # ori = self.relu(ori)
#         # ori = self.dropout(ori)
#         # ori = self.fc8(ori)
#         # ori = ori.view(-1, self.ORI_BIN_NUM, 2)
#         # ori_reg = ori[:, :, 0]
#         # ori_cls = ori[:, :, 1]

#         orientation = self.fc7(a_r_features)
#         orientation = self.relu(orientation)
#         orientation = self.dropout(orientation)
#         orientation = self.fc8(orientation)
#         # print(orientation)

#         orientation = orientation.view(-1, self.ORI_BIN_NUM, 2)
#         orientation_cls = orientation[:,:,0]
#         orientation_reg = orientation[:,:,1]

#         object_output = {'size_reg':size, 'ori_reg':orientation_reg,
#                              'ori_cls':orientation_cls, 'centroid_reg':distance_reg,
#                              'centroid_cls':distance_cls, 'offset_2D':offset}
#         return object_output

    # def forward(self, x, g_features, split, pair_counts, target):
    #     '''
    #         x: Patch x Channel x Hight x Width
    #         Geometry_feature: sum of pairs x 64
    #         split: e.g. [[0,5],[5,8]]
    #         pair_counts: e.g. [0,49,113]
    #         target: Patch x num of class: target[i]: object i should belongs to class[i], where targets[i][class[i]] =1 (?)
    #     '''
    #     a_featrues = self.resnet(x)
    #     a_features = a_featrues.view(a_featrues.size(0),-1) # N x 2048
    #     r_features = self.relnet(a_featrues, g_features, split, pair_counts) # N x 2048
    #     a_r_featrues = torch.add(a_features, r_features) # N x 2048
    #     a_r_featrues = torch.cat([a_r_featrues, target], dim = 1) # N x (2048 + class)
    #     # Use Fc to predict all we want
    #     # print(a_r_featrues)
    #     offset = self.fc_off_1(a_r_featrues)
    #     offset = self.relu_1(offset)
    #     offset = self.dropout_1(offset)
    #     offset = self.fc_off_2(offset)

    #     distance = self.fc5(a_r_featrues)
    #     distance = self.relu_1(distance)
    #     distance = self.dropout_1(distance)
    #     distance = self.fc_centroid(distance)
    #     distance = distance.view(-1, self.CENTER_BIN_NUM, 2)
    #     distance_cls = distance[:,:,0]
    #     distance_reg = distance[:,:,1]

    #     size = self.fc1(a_r_featrues)
    #     size = self.relu_1(size)
    #     size = self.dropout_1(size)
    #     size = self.fc2(size)

    
    #     orientation = self.fc3(a_r_featrues)
    #     orientation = self.relu_1(orientation)
    #     orientation = self.dropout_1(orientation)
    #     orientation = self.fc4(orientation)
    #     # print(orientation)

    #     orientation = orientation.view(-1, self.ORI_BIN_NUM, 2)
    #     orientation_cls = orientation[:,:,0]
    #     orientation_reg = orientation[:,:,1]
    #     object_output = {'size_reg':size, 'ori_reg':orientation_reg,
    #                          'ori_cls':orientation_cls, 'centroid_reg':distance_reg,
    #                          'centroid_cls':distance_cls, 'offset_2D':offset}
    #     return object_output

class ODN(nn.Module):
    def __init__(self, cfg):
        super(ODN, self).__init__()
        bin = cfg.bins
        self.ORI_BIN_NUM = len(bin['ori_bin'])
        self.CENTER_BIN_NUM = len(bin['centroid_bin'])

        self.resnet = resnet.resnet34_Half(pretrained = False)

        self.relnet = RelationNet()

        Linear_input_size = 2048 + len(NYU40CLASSES) # Depend on the dataset

        # For Projection offset (2D)
        self.fc1 = nn.Linear(Linear_input_size, 128)
        # self.fc2 = nn.Linear(128, 3)
        self.fc2 = nn.Linear(128, 2)
        
        # For distance to Camara Center, which is used to calculate Centroid of 3D Box 
        self.fc3 = nn.Linear(Linear_input_size, 128)
        self.fc4 = nn.Linear(128, 2 * self.CENTER_BIN_NUM)

        # For the size of 3D Box
        self.fc5 = nn.Linear(Linear_input_size, 128)
        self.fc6 = nn.Linear(128, 3)
        # self.fc6 = nn.Linear(128, self.OBJ_CENTER_BIN * 2)

        # For orientation of 3D box (1 dimension)
        self.fc7 = nn.Linear(Linear_input_size, 128)
        self.fc8 = nn.Linear(128, 2 * self.ORI_BIN_NUM)

        # self.relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.5)
        # Load Pretrained ResNet Model
        pretrained_dict = model_zoo.load_url(resnet.model_urls['resnet34'])
        model_dict = self.resnet.state_dict()
        pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.resnet.load_state_dict(model_dict)

        # initialize weights (I don't understand why it is necessary)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()
    def forward(self, x, g_features, split, pair_counts, target):
        '''
            x: Patch x Channel x Hight x Width
            Geometry_feature: sum of pairs x 64
            split: e.g. [[0,5],[5,8]]
            pair_counts: e.g. [0,49,113]
            target: Patch x num of class: target[i]: object i should belongs to class[i], where targets[i][class[i]] =1 (?)
        '''
        a_featrues = self.resnet(x)
        a_features = a_featrues.view(a_featrues.size(0),-1) # N x 2048
        r_features = self.relnet(a_featrues, g_features, split, pair_counts) # N x 2048
        a_r_featrues = torch.add(a_features, r_features) # N x 2048
        a_r_featrues = torch.cat([a_r_featrues, target], dim = 1) # N x (2048 + class)
        # Use Fc to predict all we want
        # print(a_r_featrues)
        offset = self.fc1(a_r_featrues)
        offset = self.relu(offset)
        offset = self.dropout(offset)
        offset = self.fc2(offset)

        distance = self.fc3(a_r_featrues)
        distance = self.relu(distance)
        distance = self.dropout(distance)
        distance = self.fc4(distance)
        distance = distance.view(-1, self.CENTER_BIN_NUM, 2)
        distance_cls = distance[:,:,0]
        distance_reg = distance[:,:,1]

        size = self.fc5(a_r_featrues)
        size = self.relu(size)
        size = self.dropout(size)
        size = self.fc6(size)

    
        orientation = self.fc7(a_r_featrues)
        orientation = self.relu(orientation)
        orientation = self.dropout(orientation)
        orientation = self.fc8(orientation)
        # print(orientation)

        orientation = orientation.view(-1, self.ORI_BIN_NUM, 2)
        orientation_cls = orientation[:,:,0]
        orientation_reg = orientation[:,:,1]
        object_output = {'size_reg':size, 'ori_reg':orientation_reg,
                             'ori_cls':orientation_cls, 'centroid_reg':distance_reg,
                             'centroid_cls':distance_cls, 'offset_2D':offset}
        return object_output