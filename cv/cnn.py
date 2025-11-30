import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

print("PyTorch Version: ", torch.__version__)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
        # 卷积层2
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)
        # 全连接层
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        # 输出层
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        """
        CNN网络结构
        :param MNIST手写数字识别数据集 x: [batch_size, 1, 28, 28]
        :return: 一张图片属于各个类别的log_softmax
        """
        # print("x size: ", x.size())

        # Step1. 卷积 + ReLU
        x = nn.functional.relu(self.conv1(x))  # [32, 20, 24, 24]
        # print("conv1 size: ", x.size())

        # Step2. 池化
        x = nn.functional.max_pool2d(x, 2, 2)  # [32, 20, 12, 12]
        # print("pool size: ", x.size())

        # Step3. 卷积 + ReLU
        x = nn.functional.relu(self.conv2(x))  # [32, 50, 8, 8]
        # print("conv2 size: ", x.size())

        # Step4. 池化
        x = nn.functional.max_pool2d(x, 2, 2)  # [32, 50, 4, 4]
        # print("pool size: ", x.size())

        # Step5. 全连接层之前的shape变化
        x = x.view(-1, 4 * 4 * 50)  # [32, 800]
        # print("view size: ", x.size())

        # Step5. 全连接层 + ReLU
        x = nn.functional.relu(self.fc1(x))  # [32, 500]
        # print("fc1 size: ", x.size())

        # Step6. 输出层softmax
        x = self.fc2(x)  # [32, 10]
        # print("fc2 size: ", x.size())

        # sys.exit()
        # 这样的返回值是为了适配 nll_loss()的输入格式
        return nn.functional.log_softmax(x, dim=1)  # [32, 10]


torch.manual_seed(53113)
batch_size = test_batch_size = 32
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# 训练集数据
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/mnist_data',
                   train=True,
                   download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),  # Step1. 将图像转换为PyTorch张量
                       transforms.Normalize((0.1307,), (0.3081,))  # Step2. 对转换后的张量进行归一化处理
                   ])),
    batch_size=batch_size,
    shuffle=True,
    **kwargs)

# 测试集数据
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/mnist_data',
                   train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size,
    shuffle=True,
    **kwargs)

# 模型
model = Net().to(device)
# 优化器
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    """
    模型训练过程
    :param model: 模型
    :param device: 设备
    :param train_loader: 数据集
    :param optimizer: 优化器
    :param epoch: 轮次
    :param log_interval: 日志间隔
    :return:
    """
    model.train()  # 训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(f"batch_idx: {batch_idx}")
        # data: [batch_size, 1, 28, 28]
        # target: [batch_size,]
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()  # 清零梯度
        output = model(data)  # 前向传播
        loss = nn.functional.nll_loss(output, target)  # 计算损失
        loss.backward()  # 反向传播, 计算梯度
        optimizer.step()  # 更新模型参数
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch}. "
                  f"Process: {batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]. "
                  f"Loss: {loss.item():.6f}")

def print_model_summary(model):
    # 打印模型结构
    print("Model structure:")
    print(model)

    # 参数总数与可训练参数数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params}, Trainable params: {trainable_params}")
    for name, param in model.named_parameters():
        pdata = param.detach().cpu()
        print(f"{name}: , shape={tuple(pdata.shape)}, numel={pdata.numel()}")

if __name__ == '__main__':
    # print_model_summary(model)
    epochs = 2
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval=100)
        
        model.eval()  # 测试模式
        test_loss = 0
        correct = 0
        with torch.no_grad():  # 不计算梯度，加快计算速度
            for data, target in test_loader:
                # data: [batch_size, 1, 28, 28]
                # target: [batch_size,]
                data, target = data.to(device), target.to(device)

                # output: [batch_size, 10]
                output = model(data)
                test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

                # pred: [batch_size, 1]
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                # print(f"target.view_as(pred).shape(): {target.view_as(pred).shape}")  # torch.Size([32, 1])

        test_loss /= len(test_loader.dataset)
        print(f'Epoch: {epoch}. '
              f'Test set: Average loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\n')

    print_model_summary(model)
    save_model = True
    if save_model:
        torch.save(model.state_dict(), "./cv/mnist_cnn.pt")