# MNIST-CNN-
just A small program


# MNIST 手写数字识别 - 卷积神经网络实现

这是一个使用PyTorch实现的卷积神经网络(CNN)，用于识别MNIST手写数字数据集。

## 项目结构

```
├── main.py           # 主程序文件，包含模型定义和训练代码
├── run_mnist.py      # 启动脚本，用于设置环境变量并运行主程序
├── requirements.txt  # 项目依赖列表
├── README.md         # 项目说明文档
├── .gitignore        # Git忽略文件配置
├── mnist/            # MNIST数据集目录
│   └── MNIST/        # PyTorch默认数据集存储路径
│       └── raw/      # 原始数据文件
└── mnist_sample.png  # 样本图像展示
```

## 功能特点

- 使用卷积神经网络(CNN)进行MNIST手写数字识别
- 支持自动下载MNIST数据集
- 包含数据可视化功能，展示样本图像
- 实现训练过程和测试评估
- 绘制测试准确率曲线

## 技术栈

- Python 3.x
- PyTorch
- NumPy
- Matplotlib

## 安装说明

1. 克隆仓库到本地

```bash
# 假设您使用git进行版本控制
# git clone <您的仓库地址>
cd MNIST
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 直接运行主程序

```bash
python main.py
```

### 使用启动脚本（解决OpenMP冲突问题）

```bash
python run_mnist.py
```

## 模型架构

该项目使用的卷积神经网络包含以下层：

1. 第一层卷积：16个5×5卷积核，ReLU激活函数，2×2最大池化
2. 第二层卷积：20个5×5卷积核，ReLU激活函数，2×2最大池化
3. 全连接层：320→50→10（输出10个类别概率）

## 训练参数

- 批次大小 (batch_size): 64
- 学习率 (learning_rate): 0.01
- 动量 (momentum): 0.5
- 训练轮数 (EPOCH): 10

## 注意事项

- 首次运行程序时，会自动下载MNIST数据集（约12MB）
- 如果遇到OpenMP运行时冲突问题，请使用`run_mnist.py`脚本启动程序
- 程序会显示12个样本图像，然后进行训练，并在每轮训练后显示测试准确率
- 训练结束后会绘制测试准确率曲线

## License

MIT
