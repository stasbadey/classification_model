import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
from keras.utils import np_utils
from keras.datasets import mnist
#kerasでCNN構築
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
import glob
from pathlib import Path





root_dir = "data/"
train_csv_filepath = root_dir + "train.csv"

# ファイルの読み込み
train_df = pd.read_csv(train_csv_filepath)

resize_w = 256
resize_h = 256
channel = 3

import cv2


# 画像が大きいと計算が遅いため、リサイズ縮小
def resize(tmp_image):
    return cv2.resize(tmp_image, (resize_h, resize_w))


# 4次元配列化()　
def to_4d(tmp_image):
    return tmp_image.reshape(1, resize_h, resize_w, channel)


# 256段階の色調を0.0~1.0にする
def normalize(tmp_image):
    return tmp_image / 255.0


# 画像の前処理付きロード
def load_preprocessed_image(image_filepath):
    tmp_image = cv2.imread(image_filepath)
    tmp_image = resize(tmp_image)
    tmp_image = normalize(tmp_image)
    tmp_image = to_4d(tmp_image)
    return tmp_image


images = None
for fn in train_df['filename']:
    image_filepath = root_dir + 'train/' + fn
    tmp_image = load_preprocessed_image(image_filepath)
    if (images is None):
        images = tmp_image
    else:
        images = np.vstack((images, tmp_image))

anomaly_flags = np.array([flag for flag in train_df['anomaly']])

# 確認
anomaly_flags = np_utils.to_categorical(anomaly_flags, 2)

model = Sequential()

model.add(Conv2D(filters=10, kernel_size=(4, 4), padding='same', input_shape=(256, 256, 3), activation='relu'))
model.add(Conv2D(filters=10, kernel_size=(3, 3), padding='same', input_shape=(64, 64, 8), activation='relu'))
model.add(Conv2D(filters=10, kernel_size=(2, 2), padding='same', input_shape=(16, 16, 16), activation='relu'))

model.add(Flatten())
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0003), metrics=['accuracy'])


X_test = images[:100]
y_test = anomaly_flags[:100]
X_train = images[100:]
y_train = anomaly_flags[100:]

history = model.fit(X_train,
                    y_train,
                    epochs=30,
                    batch_size=16,
                    verbose=1,
                    validation_data=(X_test, y_test))

train_score = model.evaluate(X_train, y_train, verbose=0)
test_score = model.evaluate(X_test, y_test, verbose=0)

print('Train Loss:{0:.3f}'.format(train_score[0]))
print('Train accuracy:{0:.3}'.format(train_score[1]))
print('Test Loss:{0:.3f}'.format(test_score[0]))
print('Test accuracy:{0:.3}'.format(test_score[1]))

test_images = None
test_filenames = None
for test_filepath in glob.glob('data/test/*.png'):
    tmp_image = load_preprocessed_image(test_filepath)
    if (test_images is None):
        test_images = tmp_image
        test_filenames = [Path(test_filepath).name]
    else:
        test_images = np.vstack((test_images, tmp_image))
        test_filenames.append(Path(test_filepath).name)

model.save(os.path.join('models', 'anotherclassifier.h5'))