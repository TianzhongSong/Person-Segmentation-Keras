import os
from tqdm import tqdm

color_img_path = '/home/deep/datasets/humanparsing/JPEGImages/'
label_img_path = '/home/deep/datasets/humanparsing/SegmentationParts/'

imgs = os.listdir(color_img_path)
limgs = os.listdir(label_img_path)

imgs.sort(key=str.lower)
limgs.sort(key=str.lower)

f1 = open('train.txt', 'w')
f2 = open('test.txt', 'w')

pd = tqdm(total=17706)
i = 0
for img, limg in zip(imgs, limgs):
    pd.update(1)
    if i >= 12706:
        f2.write('humanparsing/JPEGImages/' + img + ' ' + 'humanparsing/SegmentationParts/' + limg + '\n')
    else:
        f1.write('humanparsing/JPEGImages/' + img + ' ' + 'humanparsing/SegmentationParts/' + limg + '\n')
    i += 1
pd.close()
f1.close()
f2.close()
