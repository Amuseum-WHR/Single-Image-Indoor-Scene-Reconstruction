"""
Reference: chainercv/examples/faster_rcnn/demo.py
"""

import argparse
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

    # visualize
    vis_bbox(img, bbox, label, score, label_names=NYU40CLASSES)
    plt.show()

    return dic


def args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--pretrained-model', default='./out/2DBB/2dbb_model_80000.npz')
    parser.add_argument('--image', default='./demo/inputs/img1.jpg')
    return parser.parse_args()


if __name__ == '__main__':
    args = args('2D Bounding Boxes Demo')

    print(TwoDBB(args.gpu, args.pretrained_model, args.image))