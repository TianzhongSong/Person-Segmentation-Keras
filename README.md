# Person-Segmentation-Keras
Person segmentation with Keras (SegNet, Unet, etc.)

## Dataset

### Person segmentation

[HumanParsing-Dataset](https://github.com/lemondan/HumanParsing-Dataset) is adopted in this repo.

Origin HumanParsing-Dataset contains 16+1 object classes. But in this repo, i just segment person which is a binary classification task.

I generate new label images by my self, you can download new label imgages at https://pan.baidu.com/s/1Y6bKUznsVc7xNWb9tqWaHA passwd: p8ks

Of course you can generate label images by yourself using [convert_labels.py](https://github.com/TianzhongSong/Person-Segmentation-Keras/blob/master/convert_labels.py) , [gen_train_test_list.py](https://github.com/TianzhongSong/Person-Segmentation-Keras/blob/master/gen_train_test_list.py).

I use 12706 images of HumanParsing-Dataset as training set, the remaining images as test set.

During training, i resize images with unchanged aspect ratio using padding, for details you can see [this script](https://github.com/TianzhongSong/Person-Segmentation-Keras/blob/master/utils/segdata_generator.py).

### Human parsing

Origin HumanParsing-Dataset contains 16+1 object classes. 

    background     0
    hat            1
    hair           2 
    sunglass       3
    upper-clothes  4
    skirt          5
    pants          6
    dress          7
    belt           8
    left-shoe      9
    right-shoe     10
    face           11
    left-leg       12
    right-leg      13
    left-arm       14
    right-arm      15
    bag            16
    scarf          17

I have simplified the parsing task. Now it contains 4 + 1 classes.

    background     0 (background, bag)
    head           1 (hat, hair, sunglass, face, scarf)
    upper body     2 (upper-clothes)
    both hands     3 (left-arm, right-arm)
    lower body     4 (skirt, pants, dress, belt, left-shoe, right-shoe, left-leg, right-leg)

New label images can be downloaded from this link https://pan.baidu.com/s/1jhqpOn8oBmiJiwzohhkfww passwd: gamc

## Usage

All models are defined in 'models' directory.

An example for training Unet.

    python train_segmentation.py --model='unet'

For visiualsizing the predictions, you can use the follow command

    python predict.py --model='unet'
    
## Results

### Unet

#### Person segmentation

mIU: 0.8918

Origin images, ground truth images and predictions.

![predictions](https://github.com/TianzhongSong/Person-Segmentation-Keras/blob/master/predicts.png)

Val accuracy during training.

![val acc unet](https://github.com/TianzhongSong/Person-Segmentation-Keras/blob/master/results/unet_accuracy.png)

Val loss during training.

![val loss unet](https://github.com/TianzhongSong/Person-Segmentation-Keras/blob/master/results/unet_loss.png)

#### Human parsing

<table width="95%">
  <tr>
    <td></td>
    <td align=center><b>Part</td>
    <td align=center>mIoU</td>
  </tr>

  <tr>
    <td rowspan=5 align=center width="10%"><b>Unet</td>
    <td align=center width="10%"><b>head</td>
    <td align=center width="10%"><b>0.66476</td>
  </tr>
  <tr>
    <td align=center width="10%"><b>upper body</td>
    <td align=center width="10%"><b>0.48639</td>
  </tr>
  <tr>
    <td align=center width="10%"><b>both hands</td>
    <td align=center width="10%"><b>0.27016</td>
  </tr>
  <tr>
    <td align=center width="10%"><b>lower body</td>
    <td align=center width="10%"><b>0.66536</td>
  </tr>
  <tr>
    <td align=center width="10%"><b>mean</td>
    <td align=center width="10%"><b>0.52167</td>
  </tr>
</table>

Origin images, ground truth images and predictions.

![predictions](https://github.com/TianzhongSong/Person-Segmentation-Keras/blob/master/seg_predicts.png)

### SegNet

Todo

## Reference

[image-segmentation-keras](https://github.com/divamgupta/image-segmentation-keras)

[SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial)

