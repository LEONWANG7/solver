import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint

data = np.load('./mnist.npz')
X_train, y_train = data['x_train'], data['y_train']
X_test, y_test = data['x_test'], data['y_test']

# 调整每个图片，使值位于0到1之间
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

print('Integer-valued labes:')
print(y_train[:10])

# one-hot
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
print('One-hot labels:')
print(y_train[:10])

# 定义模型
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# 总结模型
model.summary()

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# 评估模型
score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100 * score[1]

# 打印测试的准确度
print('Test accuracy: %.4f%%' % accuracy)

# 训练模型
checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5',
                               verbose=1, save_best_only=True)
hist = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=0.2, callbacks=[checkpointer], verbose=1,
                 shuffle=True)

# 加载在验证集中最好准确度的权重
model.load_weights('mnist.model.best.hdf5')

# 在测试集中计算准确度
score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100 * score[1]
print('Test accuracy: %.4f%%' % accuracy)
