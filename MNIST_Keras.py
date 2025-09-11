import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import gzip

# 设置中文字体，确保标签能正确显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 超参数设置
batch_size = 64
learning_rate = 0.01
momentum = 0.5
EPOCH = 10

# 从本地mnist目录加载数据集
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')
    
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28, 28)
    
    return images, labels

# 加载本地数据集
print("正在从本地加载MNIST数据集...")
x_train, y_train = load_mnist('./mnist/MNIST/raw', kind='train')
x_test, y_test = load_mnist('./mnist/MNIST/raw', kind='t10k')

# 数据预处理
# 将图像数据转换为四维张量 [样本数, 高度, 宽度, 通道数]
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')

# 归一化（使用与原代码相同的均值和标准差）
mean = 0.1307
std = 0.3081
x_train = (x_train / 255.0 - mean) / std
x_test = (x_test / 255.0 - mean) / std

# 将标签转换为one-hot编码
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 测试显示
fig = plt.figure(figsize=(10, 8))
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.tight_layout()
    plt.imshow(x_train[i].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("标签: {}".format(np.argmax(y_train[i])))
    plt.xticks([])
    plt.yticks([])
plt.show()

# 使用Keras构建卷积神经网络
model = Sequential()
# 第一个卷积层
model.add(Conv2D(16, kernel_size=5, activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=2))
# 第二个卷积层
model.add(Conv2D(20, kernel_size=5, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
# 展平层
model.add(Flatten())
# 全连接层
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 编译模型
optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 训练模型
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=EPOCH,
    validation_data=(x_test, y_test),
    verbose=1
)

# 评估模型
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'测试集上的准确率为：{accuracy * 100:.2f}%')

# 绘制准确率曲线
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='测试准确率')
plt.title('模型准确率变化')
plt.xlabel('epoch')
plt.ylabel('准确率')
plt.legend()
plt.show()

# 绘制损失曲线
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='测试损失')
plt.title('模型损失变化')
plt.xlabel('epoch')
plt.ylabel('损失')
plt.legend()
plt.show()