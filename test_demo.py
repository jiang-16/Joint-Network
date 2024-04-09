import os.path

import matplotlib.pyplot as plt
from torch import nn,optim
import torch
from skimage import io
from data_CT_process import *
from net import *
from torch_CT_recon_odl import *
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn as nn
from resnet_demo import *
from loss import *




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_dataset = CTProjectionDataset(projection_dir='D:/参考文献/Joint-Network paper/proj_real/test_CBCT/real_image/proj',
                                        image_dir='D:/参考文献/Joint-Network paper/proj_real/test_CBCT/real_image/image')

    # 用DataLoader进行批处理
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
net = UNet().to(device)
resnet = MyResNet()


resnet = resnet.to(device)


net.load_state_dict(torch.load('./checkpoints/net_params.pth'))
resnet.load_state_dict(torch.load('./checkpoints/resnet_params.pth'))
net.to(device)
resnet.to(device)
net.eval()
resnet.eval()

# create the TensorRecon instance
recon_instance = TensorRecon()


# 开始测试
for i, (raw_ct_data, ideal_ct_data) in enumerate(test_dataloader):
    raw_ct_data, ideal_ct_data = raw_ct_data.to(device), ideal_ct_data.to(device)

    # 前向传播
    with torch.no_grad():  # In test phase, we don't need to compute gradients
        corrected_proj_data = net(raw_ct_data)
        recon_instance = TensorRecon()  # 创建TensorRecon实例
        reconstructed_ct = recon_instance.tensor_to_image(corrected_proj_data)
        reconstructed_ct = reconstructed_ct.to(device)
        final_ct = resnet(reconstructed_ct)
        img = final_ct.squeeze().detach().cpu().numpy()
        filename = f"{i}.png"
        io.imsave('D:/参考文献/Joint-Network paper/proj_real/test_CBCT/real_image/output'+filename, img)
        #plt.imshow(img,'gray')
        #plt.show()