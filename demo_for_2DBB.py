"""
Reference: chainercv/examples/faster_rcnn/demo.py
"""

import argparse
import os
import matplotlib.pyplot as plt

import chainer
from chainercv.links import FasterRCNNVGG16
from chainercv import utils
from chainercv.visualizations import vis_bbox

from dataset.Sunrgbd_2DBB_Dataloader import NYU40CLASSES


def TwoDBB(gpu, pretrained_model, image):
    # load model
    model = FasterRCNNVGG16(n_fg_class=len(NYU40CLASSES), pretrained_model=pretrained_model)
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    # predit
    img = utils.read_image(image, color=True)
    bboxes, labels, scores = model.predict([img])
    bbox, label, score = bboxes[0], labels[0], scores[0]
    n_objects = bbox.shape[0]
    bbox_xy = [[int(bbox[i][1]), int(bbox[i][0]), int(bbox[i][3]), int(bbox[i][2])] for i in range(n_objects)]
    dic = []
    for i in range(n_objects):
        dic.append({'bbox': bbox_xy[i], 'class': label[i]})

    return dic


def TwoDBB_demo(gpu, pretrained_model, image, visualize, save):
    # load model
    model = FasterRCNNVGG16(n_fg_class=len(NYU40CLASSES), pretrained_model=pretrained_model)
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    # predit
    img = utils.read_image(image, color=True)
    bboxes, labels, scores = model.predict([img])
    bbox, label, score = bboxes[0], labels[0], scores[0]
    n_objects = bbox.shape[0]
    bbox_xy = [[int(bbox[i][1]), int(bbox[i][0]), int(bbox[i][3]), int(bbox[i][2])] for i in range(n_objects)]
    dic = []
    for i in range(n_objects):
        dic.append({'bbox': bbox_xy[i], 'class': label[i]})

    # visualize and save
    vis_bbox(img, bbox, label, score, label_names=NYU40CLASSES)
    if save:
        plt.savefig(os.path.splitext(image)[0] + '_2DBB.jpg')
    if visualize:
        plt.show()

    return dic


def args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--pretrained-model', default='./out/2DBB/2dbb_model_80000.npz')
    parser.add_argument('--image', default='./demo/inputs/img1.jpg')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--save', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = args('2D Bounding Boxes Demo')

    if args.visualize or args.save:
        print(TwoDBB_demo(args.gpu, args.pretrained_model, args.image, args.visualize, args.save))
    else:
        print(TwoDBB(args.gpu, args.pretrained_model, args.image))
