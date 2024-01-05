import numpy as np
import torch
from torchvision import datasets, transforms
import os

# 全局变量，用于递增命名文件
global_counter = 1


def create_custom_dataset(output_folder):
    global global_counter

    # 设置随机数种子以确保可重复性
    torch.manual_seed(42)

    # 定义转换
    transform = transforms.Compose([transforms.ToTensor()])

    # 加载MNIST数据集
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    for label in range(10):
        for _ in range(60):
            # 随机选择一个图像
            index = torch.randint(0, len(mnist_dataset.targets[mnist_dataset.targets == label]), size=(1,)).item()
            selected_image = mnist_dataset.data[mnist_dataset.targets == label][index]

            random_channel = torch.randint(0, 10, size=(1,)).item()

            # 将MNIST图像放入包含10个通道的张量中
            custom_tensor = torch.zeros((10, 28, 28), dtype=torch.uint8)
            custom_tensor[random_channel] = selected_image

            # 创建输出文件
            output_file = os.path.join(output_folder, str(label))
            os.makedirs(output_file, exist_ok=True)
            output_file = os.path.join(output_file, f"{global_counter}.npy")
            global_counter += 1

            # 保存为npy文件
            np.save(output_file, custom_tensor.numpy())


if __name__ == "__main__":
    # 替换为你想保存数据集的文件夹路径
    output_folder = './mydata/finetune'

    create_custom_dataset(output_folder)
