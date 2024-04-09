import matlab.engine
import numpy as np
import scipy.io as sio
import torch
from matplotlib import pyplot as plt

class TensorRecon:
    def __init__(self):
        # 在初始化时启动MATLAB engine
        self.eng = matlab.engine.start_matlab()

    def tensor_to_image(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError('Input must be a torch.Tensor')

        # 假设 tensor 是形状为 [1, 1, m, n] 的张量
        numpy_array = tensor.squeeze().detach().cpu().numpy()
        numpy_array = numpy_array.astype(np.float32)
        proj_list = numpy_array.tolist()

        # 将 numpy 数组转换为 MATLAB 可接受的形式
        input_proj = matlab.single(proj_list)

        # 调用 MATLAB 函数 proj_to_image
        output_image = self.eng.proj_to_image(input_proj)

        # 将 MATLAB 数组转换回 numpy 数组，并转换为适当的 float 类型
        output_image = np.array(output_image)
        img = output_image.astype(np.float32)

        # 将 numpy 数组转换成 torch 张量
        img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)
        return img_tensor

# 示例使用
# tensor = your_tensor_with_shape_1_1_m_n
# output_tensor = recon_obj.tensor_to_image(tensor)

