from models import unet, segnet
import numpy as np
from tqdm import tqdm
from utils.segdata_generator import generator
import argparse


def compute_iou(gt, pt):
    intersection = np.sum(gt * pt)
    u = gt + pt
    u[u > 0] = 1
    union = np.sum(u)
    return intersection / union


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='command for training segmentation models with keras')
    parse.add_argument('--model', type=str, default='unet', help='support unet, segnet')
    parse.add_argument('--nClasses', type=int, default=2)
    parse.add_argument('--dtype', type=str, default='float32')
    args = parse.parse_args()

    n_classes = args.nClasses
    images_path = '../../datasets/segmentation/'
    val_file = './data/seg_test.txt' if n_classes == 2 else './data/parse_test.txt'
    weights_file = './weights/{}_seg_weights.h5'.format(args.model) if n_classes == 2 \
        else './weights/{}_parse_weights.h5'.format(args.model)
    input_height = 256
    input_width = 256

    if args.model == 'unet':
        m = unet.Unet(n_classes, input_height=input_height, input_width=input_width)
    elif args.model == 'segnet':
        m = segnet.SegNet(n_classes, input_height=input_height, input_width=input_width)
    else:
        raise ValueError('Do not support {}'.format(args.model))

    m.load_weights(weights_file.format(args.model))
    m.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    print('Start evaluating..')
    pbdr = tqdm(total=5000)
    iou = [0. for _ in range(1, n_classes)]
    count = [0. for _ in range(1, n_classes)]
    for x, y in generator(images_path, val_file, 1, n_classes, input_height, input_width, train=False):
        pbdr.update(1)
        pr = m.predict(x)[0]
        pt = pr.argmax(axis=1)
        for c in range(1, n_classes):
            gt = y[:, :, c]
            gt = gt.reshape((input_height * input_width))
            gt_img = np.zeros_like(y)
            pt_img = np.zeros_like(y)
            gt_img[:] += (gt[:] == 1).astype('uint8')
            pt_img[:] += (pt[:] == c).astype('uint8')
            if not (pt_img == np.zeros_like(pt_img)).all() or not (gt_img == np.zeros_like(gt_img)).all() :
                iou[c - 1] += compute_iou(pt_img, gt_img)
                count[c - 1] += 1
    miou = 0.
    for c in range(1, n_classes):
        m = iou[c - 1] / count[c - 1]
        miou += m
        print('mIoU: class {0}: {1}'.format(c, m))
    print('mIoU:{}'.format(miou / (n_classes - 1)))
    pbdr.close()
