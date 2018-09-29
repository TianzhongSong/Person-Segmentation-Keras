import glob
import numpy as np
import cv2
import random
import argparse


def imageSegmentationGenerator(images_path, segs_path, n_classes):
    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'

    images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg")
    images.sort()
    segmentations = glob.glob(segs_path + "*.jpg") + glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg")
    segmentations.sort()

    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n_classes)]
    # colors = np.array([[0, 0, 0], [255, 255, 255]])

    assert len(images) == len(segmentations)

    for im_fn, seg_fn in zip(images, segmentations):
        assert (im_fn.split('/')[-1].split('.')[0] == seg_fn.split('/')[-1].split('.')[0])

        img = cv2.imread(im_fn)
        seg = cv2.imread(seg_fn)
        print(np.unique(seg))

        seg_img = np.zeros_like(seg)

        for c in range(n_classes):
            seg_img[:, :, 0] += ((seg[:, :, 0] == c) * (colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((seg[:, :, 0] == c) * (colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((seg[:, :, 0] == c) * (colors[c][2])).astype('uint8')

        cv2.imshow("img", img)
        cv2.imshow("seg_img", seg_img)
        cv2.waitKey(100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, default='/home/deep/datasets/humanparsing/JPEGImages/')
    parser.add_argument("--annotations", type=str, default='/home/deep/datasets/humanparsing/Segmentations/')
    parser.add_argument("--n_classes", type=int, default=2)
    args = parser.parse_args()

    imageSegmentationGenerator(args.images, args.annotations, args.n_classes)
