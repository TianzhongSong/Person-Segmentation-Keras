from models import unet, segnet
import cv2
import numpy as np
import argparse
from utils.segdata_generator import generator


def predict_segmentation():
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

    colors = np.array([[0, 0, 0], [255, 255, 255]])
    i = 0
    for x, y in generator(images_path, val_file, 1, n_classes, input_height, input_width):
        pr = m.predict(x)[0]
        pr = pr.reshape((input_height, input_width, n_classes)).argmax(axis=2)
        seg_img = np.zeros((input_height, input_width, 3))
        for c in range(n_classes):
            seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
        cv2.imshow('test', seg_img)
        cv2.imwrite('./output/{}.jpg'.format(i), seg_img)
        i += 1
        cv2.waitKey(30)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='command for training segmentation models with keras')
    parse.add_argument('--model', type=str, default='unet', help='support unet, segnet')
    args = parse.parse_args()
    predict_segmentation()
