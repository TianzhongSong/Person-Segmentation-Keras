from models import unet, segnet
import numpy as np
from tqdm import tqdm
from utils.segdata_generator import generator
import argparse


def compute_iou(gt, pt):
    intersection = 0
    union = 0
    for i, j in zip(gt, pt):
        if i == 1 or j == 1:
            union += 1
        if i == 1 and j == 1:
            intersection += 1

    return intersection / union


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='command for training segmentation models with keras')
    parse.add_argument('--model', type=str, default='unet', help='support unet, segnet')
    args = parse.parse_args()

    n_classes = 2
    images_path = '/home/deep/datasets/'
    val_file = './data/seg_test.txt'
    input_height = 256
    input_width = 256

    if args.model == 'unet':
        m = unet.Unet(n_classes, input_height=input_height, input_width=input_width)
    elif args.model == 'segnet':
        m = segnet.SegNet(n_classes, input_height=input_height, input_width=input_width)
    else:
        raise ValueError('Do not support {}'.format(args.model))

    m.load_weights("./results/{}_weights.h5".format(args.model))
    m.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    print('Start evaluating..')
    pbdr = tqdm(total=5000)
    iou = 0.
    for x, y in generator(images_path, val_file, 1, n_classes, input_height, input_width, train=False):
        pbdr.update(1)
        pr = m.predict(x)[0]
        pr = pr.reshape((input_height, input_width, n_classes)).argmax(axis=2)
        pt = pr.reshape((input_height * input_width))
        y = y[:, :, 1]
        gt = y.reshape((input_height * input_width))
        iou += compute_iou(gt, pt)
    pbdr.close()
    print('mean iou:{}'.format(iou / 5000))
