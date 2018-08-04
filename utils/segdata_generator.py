import numpy as np
import cv2
import random
import sys

sys.path.append('../')


def get_batch(items, root_path, nClasses, height, width):
    x = []
    y = []
    for item in items:
        image_path = root_path + item.split(' ')[0]
        label_path = root_path + item.split(' ')[-1].strip()
        img = cv2.imread(image_path, 1)
        label_img = cv2.imread(label_path, 1)
        im = np.zeros((height, width, 3), dtype='uint8')
        im[:, :, :] = 128
        lim = np.zeros((height, width, 3), dtype='uint8')

        if img.shape[0] >= img.shape[1]:
            scale = img.shape[0] / height
            new_width = int(img.shape[1] / scale)
            diff = (width - new_width) // 2
            img = cv2.resize(img, (new_width, height))
            label_img = cv2.resize(label_img, (new_width, height))

            im[:, diff:diff + new_width, :] = img
            lim[:, diff:diff + new_width, :] = label_img
        else:
            scale = img.shape[1] / width
            new_height = int(img.shape[0] / scale)
            diff = (height - new_height) // 2
            img = cv2.resize(img, (width, new_height))
            label_img = cv2.resize(label_img, (width, new_height))
            im[diff:diff + new_height, :, :] = img
            lim[diff:diff + new_height, :, :] = label_img
        lim = lim[:, :, 0]
        seg_labels = np.zeros((height, width, nClasses))
        for c in range(nClasses):
            seg_labels[:, :, c] = (lim == c).astype(int)
        im = np.float32(im) / 127.5 - 1
        seg_labels = np.reshape(seg_labels, (width * height, nClasses))
        x.append(im)
        y.append(seg_labels)
    return x, y


def generator(root_path, path_file, batch_size, n_classes, input_height, input_width, train=True):
    f = open(path_file, 'r')
    items = f.readlines()
    f.close()
    while True:
        shuffled_items = []
        index = [n for n in range(len(items))]
        random.shuffle(index)
        for i in range(len(items)):
            shuffled_items.append(items[index[i]])
        for j in range(len(items) // batch_size):
            x, y = get_batch(shuffled_items[j * batch_size:(j + 1) * batch_size],
                             root_path, n_classes, input_height, input_width)
            yield np.array(x), np.array(y)
