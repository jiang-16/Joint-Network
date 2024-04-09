from utilis import *
import numpy as np
import os
import glob
from scipy.io import loadmat
from pathlib import Path
import torch
from matplotlib import pyplot as plt
from skimage import io



class TensorRecon:
    def __init__(self):
        self.geo = {
            "DSD": 1500,
            "DSO": 500,
            "nDetector": np.array([704, 1]),
            "dDetector": np.array([1, 1]) * 0.3,
            "sDetector": None,
            "nVoxel": np.array([512, 512]) / 1,
            "dVoxel": np.array([1, 1]) * 0.1,
            "sVoxel": None,
            "detoffset": np.array([0, 0]),
            "orgoffset": np.array([0, 0, 0])
        }

    def tensor_to_image(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError('Input must be a torch.Tensor')

        # 假设 tensor 是形状为 [1, 1, m, n] 的张量
        numpy_array = tensor.squeeze().detach().cpu().numpy()
        numpy_array = numpy_array.astype(np.float32)

        reconstructor = CTReconstruction(numpy_array, self.geo)
        fbp = reconstructor.reconstruct()

        # 将 MATLAB 数组转换回 numpy 数组，并转换为适当的 float 类型
        output_image = np.array(fbp)
        img = output_image.astype(np.float32)
        img_np_clipped = np.clip(img, 0, None)


        # 找到最大值和最小值
        min_val = np.min(img_np_clipped)
        max_val = np.max(img_np_clipped)

        # 按照 (x - min) / (max - min) 的公式进行归一化到 [0, 1]
        img_np_normalized = (img_np_clipped - min_val) / (max_val - min_val) * 255


        # 将 numpy 数组转换成 torch 张量
        img_tensor = torch.tensor(img_np_normalized,dtype=torch.float32, requires_grad=True).unsqueeze(0).unsqueeze(0)
        return img_tensor

# 示例使用
# tensor = your_tensor_with_shape_1_1_m_n
# output_tensor = recon_obj.tensor_to_image(tensor)
if __name__=='__main__':

    recon = TensorRecon()

    mat_files_dir = './Test3/proj_ideal'
    save_images_dir = 'D:/参考文献/Joint-Network paper/Test/input/used/'
    i = 0

    # 遍历指定目录下所有的.mat文件
    for file in os.listdir(mat_files_dir):
        # 确认文件是一个.mat文件
        if file.endswith('.mat'):
            # 构建完整的文件路径
            file_path = os.path.join(mat_files_dir, file)
            # 加载.mat文件
            i=i+1



        # 加载.mat文件
            data = loadmat(file_path)
            projections = data['proj1']
            proj_np = np.array(projections).astype(np.float32)
            #save_proj = os.path.join(save_proj, f'fig_{i}.png')
            #io.imsave(save_proj, proj_np)

        # 转换至PyTorch Tensor并增加所需的维度
            proj = torch.FloatTensor(proj_np).unsqueeze(0).unsqueeze(0)

        # 使用TensorRecon类进行图像重建
            img = recon.tensor_to_image(proj)
            img = img.detach().cpu().numpy()  # 确保放在cpu上，并转为numpy array
            img = img.squeeze()

        # 保存图像
        # 提取文件编号和构建保存路径
        # 提取文件编号，例如：'0030'
            save_path = os.path.join(save_images_dir, f'fig_{i}.png')
            io.imsave(save_path, img)

            print(f'Image saved: {save_path}')