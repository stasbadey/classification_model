from keras.models import load_model
import tensorflow as tf
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np


model = load_model(os.path.join('models', 'lowclassifier.h5'))

while True:
    print('Input name of test image:')
    name = input()

    img = cv2.imread('data/test/test_'+name+'.png')
    plt.imshow(img)

    resize = tf.image.resize(img, (256,256))
    plt.imshow(resize.numpy().astype(int))
    plt.axis('off')
    plt.show()
    yhat = model.predict(np.expand_dims(resize/255, 0))
    print(yhat)

    if yhat > 0.7:
        print(f'Predicted class is Normal')
    else:
        print(f'Predicted class is Defected')