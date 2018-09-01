import os
import cv2
from tqdm import tqdm
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--img-path', type=str, default='')
parse.add_argument('--save-path', type=str, default='')
parse.add_argument('--mode', type=str, default='segmentation')

args = parse.parse_args()

data_path = args.img_path
save_path = args.save_path

limgs = os.listdir(data_path)

pd = tqdm(total=17706)
if args.mode == 'segmentation':
    for item in limgs:
        pd.update(1)
        im = cv2.imread(data_path + item)
        im[im != 0] = 1

        cv2.imwrite(save_path + item, im)
else:
    for item in limgs:
        pd.update(1)
        im = cv2.imread(data_path + item)
        im[im == 1] = 1
        im[im == 2] = 1
        im[im == 3] = 1
        im[im == 17] = 1
        im[im == 11] = 1

        im[im == 4] = 2

        im[im == 14] = 3
        im[im == 15] = 3

        im[im == 5] = 4
        im[im == 6] = 4
        im[im == 7] = 4
        im[im == 8] = 4
        im[im == 9] = 4
        im[im == 10] = 4
        im[im == 12] = 4
        im[im == 13] = 4
        im[im == 16] = 0

        cv2.imwrite(save_path + item, im)

pd.close()
