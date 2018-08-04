from models import unet, segnet
from utils import segdata_generator
from keras.optimizers import Adam, SGD
from keras import backend as K
from utils.schedules import onetenth_20_30
from keras.callbacks import ModelCheckpoint
import os
import argparse
import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt

K.clear_session()


def plot_history(history, result_dir, prefix):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, '{}_accuracy.png'.format(prefix)))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, '{}_loss.png'.format(prefix)))
    plt.close()


def save_history(history, result_dir, prefix):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, '{}_result.txt'.format(prefix)), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()


def main():
    nClasses = 2
    train_batch_size = 16
    val_batch_size = 16
    init_lr = 0.01
    epochs = 50
    img_height = 256
    img_width = 256
    root_path = '/home/deep/datasets/'
    train_file = './data/seg_train.txt'
    val_file = './data/seg_test.txt'
    if args.model == 'unet':
        model = unet.Unet(nClasses, input_height=img_height, input_width=img_width)
    elif args.model == 'segnet':
        model = segnet.SegNet(nClasses, input_height=img_height, input_width=img_width)
    else:
        raise ValueError('Does not support {}, only supports unet and segnet now'.format(args.model))

    sgd = SGD(lr=init_lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-4),
                  metrics=['accuracy'])
    model.summary()


    train = segdata_generator.generator(root_path, train_file, train_batch_size, nClasses, img_height, img_width)

    val = segdata_generator.generator(root_path, val_file, val_batch_size, nClasses, img_height, img_width, train=False)

    checkpoint = ModelCheckpoint('./results/{}_weights.h5'.format(args.model),
                                 monitor='val_acc', save_best_only=True, save_weights_only=True, verbose=1)
    history = model.fit_generator(train,
                        steps_per_epoch=12706 // train_batch_size,
                        validation_data=val,
                        validation_steps=5000 // val_batch_size,
                        epochs=epochs,
                        callbacks=[checkpoint],
                        # callbacks=[onetenth_20_30(init_lr), checkpoint]
                                  )
    plot_history(history, './results/', args.model)
    save_history(history, './results/', args.model)
    if not os.path.exists('./results/'):
        os.mkdir('./results')
    model.save_weights('./results/{}_weights.h5'.format(args.model))


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='command for training segmentation models with keras')
    parse.add_argument('--model', type=str, default='segnet', help='support unet, segnet')
    args = parse.parse_args()

    main()
