FROM tensorflow/tensorflow

RUN mkdir -p /data/src && pip install keras opencv-python
RUN apt-get update && apt-get install -y libsm6 libxext6 libfontconfig1 libxrender1

RUN mkdir -p /data/src/Person-Segmentation-Keras
WORKDIR /data/src/Person-Segmentation-Keras

CMD ['python train_segmentation.py --model='unet'']
