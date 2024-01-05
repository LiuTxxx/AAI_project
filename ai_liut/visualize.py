import numpy as np
import matplotlib.pyplot as plt
import os


def visualize_npy_files(folder_path):
    # 获取文件夹中的所有.npy文件
    npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

    for npy_file in npy_files:
        # 读取npy文件
        file_path = os.path.join(folder_path, npy_file)
        data = np.load(file_path)

        # 寻找包含数字的通道
        digit_channel = None
        for i in range(data.shape[0]):
            if np.any(data[i] != 0):
                digit_channel = i
                break

        if digit_channel is not None:
            # 提取数字通道的数据
            digit_data = data[digit_channel]

            # 灰度可视化
            plt.imshow(digit_data, cmap='gray')
            plt.title(f'Digit Channel in {npy_file}')
            plt.show()


if __name__ == "__main__":
    # 替换为你的文件夹路径
    folder_path = './mydata/val/1'
    visualize_npy_files(folder_path)
