import numpy as np
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.datasets import load_files
from keras.utils import np_utils
from extract_bottleneck_features import extract_Xception
from keras.preprocessing import image
from tqdm import tqdm
from glob import glob


# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


def path_to_tensor(img_path):
    # 用PIL加载RGB图像为PIL.Image.Image类型
    img = image.load_img(img_path, target_size=(224, 224))
    # 将PIL.Image.Image类型转化为格式为(224, 224, 3)的3维张量
    x = image.img_to_array(img)
    # 将3维张量转化为格式为(1, 224, 224, 3)的4维张量并返回
    return np.expand_dims(x, axis=0)


# load train, test, and validation datasets
train_files, train_targets = load_dataset('/data/dog_images/train')
valid_files, valid_targets = load_dataset('/data/dog_images/valid')
test_files, test_targets = load_dataset('/data/dog_images/test')
# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("/data/dog_images/train/*/"))]

# 从另一个预训练的CNN获取bottleneck特征
bottleneck_features = np.load('/data/bottleneck_features/DogXceptionData.npz')
train_Xception = bottleneck_features['train']
valid_Xception = bottleneck_features['valid']
test_Xception = bottleneck_features['test']

# 定义框架
print(train_Xception.shape[1:])
Xception_model = Sequential()
Xception_model.add(GlobalAveragePooling2D(input_shape=train_Xception.shape[1:]))
Xception_model.add(Dropout(0.21))
Xception_model.add(Dense(133, activation='softmax'))
print(Xception_model.summary())

# 编译模型
Xception_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
checkpointer = ModelCheckpoint(filepath='./weights.best.Xception.hdf5', verbose=1, save_best_only=True)
Xception_model.fit(train_Xception, train_targets, validation_data=(valid_Xception, valid_targets), epochs=15,
                   batch_size=200, callbacks=[checkpointer], verbose=1)

# 加载具有最佳验证loss的模型权重
Xception_model.load_weights('./weights.best.Xception.hdf5')

# 在测试集上计算分类准确率
Xception_predictions = [np.argmax(Xception_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Xception]
test_accuracy = 100 * np.sum(np.array(Xception_predictions) == np.argmax(test_targets, axis=1)) / len(
    Xception_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


# 写一个函数，该函数将图像的路径作为输入
# 然后返回此模型所预测的狗的品种
def Xception_predict_breed(img_path):
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    predicted_vector = Xception_model.predict(bottleneck_feature)
    return dog_names[np.argmax(predicted_vector)]


# test
for path in glob("mydog/*"):
    print('图片：', path)
    print(print(Xception_predict_breed(path)))
    print('---------------------')
