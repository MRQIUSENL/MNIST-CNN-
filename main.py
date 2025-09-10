import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

batch_size = 64
learning_rate = 0.01
momentum = 0.5
EPOCH = 10

#归一化
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#获取数据集
train_dataset = datasets.MNIST(root='./mnist', train=True, download=True, transform=transform)  
test_dataset = datasets.MNIST(root='./mnist', train=False, download=True, transform=transform)  # train=True训练集，=False测试集
#测试显示
fig = plt.figure()
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.tight_layout()
    plt.imshow(train_dataset.data[i].numpy(), cmap='gray', interpolation='none')
    plt.title("Labels: {}".format(train_dataset.targets[i].item()))
    plt.xticks([])
    plt.yticks([])
plt.show()


#创建卷积神经网络

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,20,kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320,50),
            torch.nn.Linear(50,10),
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)  # 使用x.size(0)代替固定的batch_size，更灵活
        x = self.fc(x)
        return x

model = Net()

#交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss()
#优化器
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum)

#封装

def train(epoch):
    running_loss = 0.0
    running_total = 0
    running_correct = 0

    for batch_idx,data in enumerate(train_loader,0):
        inputs,target = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _,predicted = torch.max(outputs.data,dim=1)
        running_total += target.size(0)
        running_correct += (predicted == target).sum().item()
        if batch_idx % 300 == 299:
            print('[%d,%5d] loss:%.3f' % (epoch+1,batch_idx+1,running_loss/300))
            running_loss = 0.0
            running_total=0
            running_correct=0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print('测试集上的准确率为：%d %%' % (100*acc))
    return acc

if __name__ == '__main__':
    # 初始化准确率列表
    acc_list_test = []
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    for epoch in range(EPOCH):
        train(epoch)
        acc_test = test()
        acc_list_test.append(acc_test)
    
    plt.plot(acc_list_test,label='test')
    plt.legend()
    plt.show()