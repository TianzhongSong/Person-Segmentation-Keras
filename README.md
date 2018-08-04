# Person-Segmentation-Keras
Person segmentation with Keras (SegNet, Unet, etc.)

### Dataset

[HumanParsing-Dataset](https://github.com/lemondan/HumanParsing-Dataset) is adopted in this repo.

Origin HumanParsing-Dataset contains 16+1 object classes. But in this repo, i just segment person which is a binary classification task.

I generate new label images by my self, you can download new label imgages at https://pan.baidu.com/s/1Y6bKUznsVc7xNWb9tqWaHA passwd: p8ks

I use 12706 images of HumanParsing-Dataset as training set, the remaining images as test set.

During training, i resize images with unchanged aspect ratio using padding, for details you can see [this script](https://github.com/TianzhongSong/Person-Segmentation-Keras/blob/master/utils/segdata_generator.py).

### Usage

All models are defined in 'models' directory.

An example for training Unet.

    python train_segmentation.py --model='unet'

For evaluating you can use the follow command,

    python predict.py --model='unet'
    
### Results

#### Unet

Origin images, ground truth images and predictions.

![predictions](https://github.com/TianzhongSong/Person-Segmentation-Keras/blob/master/predicts.png)

Val accuracy during training.

![val acc unet](https://github.com/TianzhongSong/Person-Segmentation-Keras/blob/master/results/unet_accuracy.png)

Val loss during training.

![val loss unet](https://github.com/TianzhongSong/Person-Segmentation-Keras/blob/master/results/unet_loss.png)

#### SegNet

Todo
