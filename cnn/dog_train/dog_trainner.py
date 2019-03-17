from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
import random
import cv2
from detect_face import detect
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from tqdm import tqdm
from PIL import ImageFile

random.seed(8675309)


# 【导入数据集】
# 导入狗狗
# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


# load train, test, and validation datasets
train_files, train_targets = load_dataset('/data/dog_images/train')
valid_files, valid_targets = load_dataset('/data/dog_images/valid')
test_files, test_targets = load_dataset('/data/dog_images/test')
# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("/data/dog_images/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.' % len(test_files))

# 导入人脸
# 加载打乱后的人脸数据集的文件名
human_files = np.array(glob("/data/lfw/*/*"))
random.shuffle(human_files)

# 打印数据集的数据量
print('There are %d total human images.' % len(human_files))

# 【检测人脸】
# detect(human_files[3])

face_cascade = cv2.CascadeClassifier('/data/haarcascades/haarcascade_frontalface_alt.xml')


# 人脸识别器
# 如果img_path路径表示的图像检测到了脸，返回"True"
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


human_files_short = human_files[:100]
dog_files_short = train_files[:100]
## 请不要修改上方代码


## 基于human_files_short和dog_files_short中的图像测试face_detector的表现
# face_detected_in_human = sum([face_detector(filepath) for filepath in human_files_short])
# face_detected_in_dog = sum([face_detector(filepath) for filepath in dog_files_short])
#
# print(f'human_files 的前100张图像中，能够检测到人脸的图像占比：{face_detected_in_human / 100}')
# print(f'dog_files 的前100张图像中，能够检测到人脸的图像占比：{face_detected_in_dog / 100}')

# 【检测狗狗】
# 定义ResNet50模型
ResNet50_model = ResNet50(weights='imagenet')


# 数据预处理
def path_to_tensor(img_path):
    # 用PIL加载RGB图像为PIL.Image.Image类型
    img = image.load_img(img_path, target_size=(224, 224))
    # 将PIL.Image.Image类型转化为格式为(224, 224, 3)的3维张量
    x = image.img_to_array(img)
    # 将3维张量转化为格式为(1, 224, 224, 3)的4维张量并返回
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = (path_to_tensor(img_path) for img_path in tqdm(img_paths))
    return np.vstack(list_of_tensors)


# 基于 ResNet-50 架构进行预测
def ResNet50_predict_labels(img_path):
    # 返回img_path路径的图像的预测向量
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


# 完成狗检测模型
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


# 评估狗狗检测模型
### 测试dog_detector函数在human_files_short和dog_files_short的表现
# dog_detected_in_human = sum([dog_detector(filepath) for filepath in human_files_short])
# print(f'human_files_short中图像检测到狗狗的百分比是：{dog_detected_in_human / 100}')
# dog_detected_in_dog = sum([dog_detector(filepath) for filepath in dog_files_short])
# print(f'dog_files_short中图像检测到狗狗的百分比是：{dog_detected_in_dog / 100}')

# 【从头开始创建一个CNN来分类狗品种】
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Keras中的数据预处理过程
train_tensors = paths_to_tensor(train_files).astype('float32') / 255
print(len(valid_files))
valid_tensors = paths_to_tensor(valid_files).astype('float32') / 255
# test_tensors = paths_to_tensor(test_files).astype('float32') / 255

# model = Sequential()
#
# ### 定义你的网络架构
# model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(224, 224, 3)))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(GlobalAveragePooling2D(data_format='channels_last'))
# model.add(Dense(133, activation='softmax'))
# model.summary()
# # 编译模型
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#
# from keras.callbacks import ModelCheckpoint
#
# ### TODO: 设置训练模型的epochs的数量
#
# epochs = 5
#
# checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5',
#                                verbose=1, save_best_only=True)
#
# model.fit(train_tensors, train_targets,
#           validation_data=(valid_tensors, valid_targets),
#           epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
