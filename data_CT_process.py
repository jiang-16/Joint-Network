# DataProcess.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import scipy.io as sio
from PIL import Image
import os
import numpy as np



transform = transforms.Compose([
    transforms.ToTensor(),
])
class CTProjectionDataset(Dataset):
    def __init__(self, projection_dir, image_dir):
        """
        Args:
            projection_dir (string): Directory with all the projection .mat files.
            image_dir (string): Directory with all the corresponding CT images in .png format.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.projection_dir = projection_dir
        self.image_dir = image_dir
        self.transform = transform
        # 检索所有文件名
        self.projections = [f for f in os.listdir(projection_dir) if f.endswith('.mat')]
        self.images = [f.replace('.mat', '.png') for f in self.projections]

    def __len__(self):
        return len(self.projections)

    def __getitem__(self, idx):
        # 加载.mat文件
        mat_file_path = os.path.join(self.projection_dir, self.projections[idx])
        mat_data = sio.loadmat(mat_file_path)
        projection = mat_data['proj1']  # 替换'whateverKeyYouHave'为.mat文件中实际的键
        proj_np = np.array(projection)
        proj_np = proj_np.astype(np.float32)  # 将数据转化为float32类型
        # 将小于0的值设置为0
        proj_np_clipped = np.clip(proj_np, 0, None)

        # 找到最大值和最小值
        min_val = np.min(proj_np_clipped)
        max_val = np.max(proj_np_clipped)

        # 按照 (x - min) / (max - min) 的公式进行归一化到 [0, 1]
        proj_np_normalized = (proj_np_clipped - min_val) / (max_val - min_val)

        # 加载对应的CT图像
        image_file_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(image_file_path)
        image = image.convert('L')  # 确保为灰度图
        img_np = np.array(image)
        img_np =img_np.astype(np.float32)
        img_np_clipped = np.clip(img_np, 0, None)

        # 找到最大值和最小值
        min_val = np.min(img_np_clipped)
        max_val = np.max(img_np_clipped)

        # 按照 (x - min) / (max - min) 的公式进行归一化到 [0, 1]
        img_np_normalized = (img_np_clipped - min_val) / (max_val - min_val) * 255





        proj_tensor = torch.FloatTensor(proj_np_normalized).unsqueeze(0)  # 增加channel维度
        img_tensor = torch.FloatTensor(img_np_normalized).unsqueeze(0)  # 增加channel维度




        return proj_tensor,img_tensor

# 对CT图像进行必要的转换

if __name__=='__main__':
# 创建用于训练的Dataset实例
# 假设 'projections' 和 'images' 分别是保存.mat文件和.png文件的目录
    train_dataset = CTProjectionDataset(projection_dir='D:\\file_jiang\\Proj\\proj_demo', image_dir='D:\\file_jiang\\Proj\\img_demo')
    print(train_dataset)
# 用DataLoader进行批处理
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    for i, (ct, seg) in enumerate(train_dataloader):
        print(i, ct.size(), seg.size())



