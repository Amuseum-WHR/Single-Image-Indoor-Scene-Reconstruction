"""
Reference: chainercv/examples/faster_rcnn/train.py
"""

import argparse
import datetime
import matplotlib
import numpy as np
import os
import os.path as osp
import sys

import chainer
from chainer import training
from chainer.datasets import TransformDataset
from chainer.training import extensions
from chainercv import transforms
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links import FasterRCNNVGG16
from chainercv.links.model.faster_rcnn import FasterRCNNTrainChain

from dataset.Sunrgbd_2DBB_Dataloader import SunDataset_2DBB

sys.path.append(osp.curdir)
matplotlib.use('Agg')

sunrgbd_bbox_label_names = ('void',
                            'wall', 'floor', 'cabinet', 'bed', 'chair',
                            'sofa', 'table', 'door', 'window', 'bookshelf',
                            'picture', 'counter', 'blinds', 'desk', 'shelves',
                            'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat',
                            'clothes', 'ceiling', 'books', 'refridgerator', 'television',
                            'paper', 'towel', 'shower_curtain', 'box', 'whiteboard',
                            'person', 'night_stand', 'toilet', 'sink', 'lamp',
                            'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop')


class Transform(object):

    def __init__(self, faster_rcnn):
        self.faster_rcnn = faster_rcnn

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = self.faster_rcnn.prepare(img)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = transforms.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = transforms.random_flip(img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale


def args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--root_path', '-path', type=str, default=".")
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--lr', '-l', type=float, default=0.001)
    parser.add_argument('--out', '-o', default='./out', help='Output directory')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--step_size', '-ss', type=int, default=25000)
    parser.add_argument('--iteration', '-i', type=int, default=100000)
    parser.add_argument('--lr_shift', '-ls', type=float, default=0.5)
    parser.add_argument('--save_interval', '-si', type=int, default=5000)
    parser.add_argument('--evaluation_interval', '-ei', type=int, default=5000)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = args('Training Faster R-CNN for 2D Bounding Boxes on Sun RGB-D')

    np.random.seed(args.seed)
    now_time = str(datetime.datetime.now().replace(microsecond=0)).replace(' ', '_').replace(':', '-')
    save_dir = osp.join(args.out, now_time)
    os.mkdir(save_dir)

    # get data
    train_data = SunDataset_2DBB(args.root_path, mode="train")
    test_data = SunDataset_2DBB(args.root_path, mode="test")

    # get model
    faster_rcnn = FasterRCNNVGG16(n_fg_class=len(sunrgbd_bbox_label_names), pretrained_model='imagenet')
    faster_rcnn.use_preset('evaluate')
    model = FasterRCNNTrainChain(faster_rcnn)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # get iterators
    train_data = TransformDataset(train_data, Transform(faster_rcnn))
    train_iter = chainer.iterators.MultiprocessIterator(train_data, batch_size=1, shared_mem=100000000)
    test_iter = chainer.iterators.SerialIterator(test_data, batch_size=1, repeat=False, shuffle=False)

    # define optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(rate=0.0005))
    updater = chainer.training.updaters.StandardUpdater(train_iter, optimizer, device=args.gpu)

    # define trainer
    trainer = training.Trainer(updater, (args.iteration, 'iteration'), out=save_dir)
    trainer.extend(extensions.ExponentialShift('lr', args.lr_shift), trigger=(args.step_size, 'iteration'))

    # save log
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(chainer.training.extensions.observe_lr(), trigger=(20, 'iteration'))
    trainer.extend(extensions.LogReport(trigger=(20, 'iteration')))
    trainer.extend(extensions.PlotReport(['main/loss'], file_name='loss.png', trigger=(100, 'iteration')),
                   trigger=(100, 'iteration'))

    # save model
    trainer.extend(extensions.snapshot_object(model.faster_rcnn, '2dbb_model_{.updater.iteration}.npz'),
                   trigger=(args.save_interval, 'iteration'))

    # print
    trainer.extend(extensions.PrintReport(['iteration', 'epoch', 'elapsed_time', 'lr', 'main/loss',
                                           'main/roi_loc_loss', 'main/roi_cls_loss',
                                           'main/rpn_loc_loss', 'main/rpn_cls_loss',
                                           'validation/main/map']), trigger=(20, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # test
    trainer.extend(DetectionVOCEvaluator(test_iter, model.faster_rcnn, use_07_metric=False,
                                         label_names=sunrgbd_bbox_label_names),
                   trigger=(args.evaluation_interval, 'iteration'))

    trainer.run()
