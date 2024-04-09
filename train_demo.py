import os
from PIL import Image
from torch import nn,optim
import torch
from data_CT_process import *
from net import *
from torch_CT_recon_odl import *
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn as nn
from resnet_demo import *
from loss import *
import matplotlib.pyplot as plt

# 脚本的其余部分


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = CTProjectionDataset(projection_dir='./Train/proj',
                                        image_dir='./Train/img')

    # 用DataLoader进行批处理
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
net = UNet().to(device)
# 加载预训练权重
pretrained_weights = torch.load('./Pretrained/net_params.pth')
net.load_state_dict(pretrained_weights)


resnet = MyResNet()


resnet = resnet.to(device)
#criterion = nn.MSELoss()
criterion = CombinedLoss(weights=[0.8, 0.1, 0.1])
optimizer = optim.Adam(list(net.parameters()) + list(resnet.parameters()), lr=0.0001)
#optimizer = optim.Adam(list(net.parameters()), lr=0.0001)
num_epochs = 500
for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0
    for batch_idx, (raw_ct_data, ideal_ct_data) in enumerate(train_dataloader):
            raw_ct_data, ideal_ct_data =raw_ct_data.to(device),ideal_ct_data.to(device)
            # 初始化梯度
            optimizer.zero_grad()

            # 前向传播
            corrected_proj_data = net(raw_ct_data)
            recon_instance = TensorRecon()  # 创建TensorRecon实例
            reconstructed_ct = recon_instance.tensor_to_image(corrected_proj_data)
            reconstructed_ct = reconstructed_ct.to(device)
            final_ct = resnet(reconstructed_ct)

            #计算损失
            loss = criterion(final_ct, ideal_ct_data)
            epoch_loss += loss.item()
            num_batches += 1
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            if batch_idx == len(train_dataloader) - 2:
                img_CT = final_ct.detach().cpu().numpy()
                img_CT = img_CT.squeeze()
                img_np = img_CT.astype(np.float32)
                img_np_clipped = np.clip(img_np, 0, None)

                # 找到最大值和最小值
                min_val = np.min(img_np_clipped)
                max_val = np.max(img_np_clipped)

                # 按照 (x - min) / (max - min) 的公式进行归一化到 [0, 1]
                img_np_normalized = (img_np_clipped - min_val) / (max_val - min_val) * 255



                ideal_CT = ideal_ct_data.detach().cpu().numpy()
                ideal_CT = ideal_CT.squeeze()
                combined_image = np.concatenate((img_np_normalized, ideal_CT), axis=1)
                plt.imsave(os.path.join('./epoch/', f'epoch_{epoch + 1}_batch_{batch_idx + 1}.png'),combined_image)





            # 计算平均损失
    average_loss = epoch_loss / num_batches
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {average_loss}')
    if (epoch + 1) % 100 == 0:
        torch.save(net.state_dict(), os.path.join('./checkpoints/', f'net_params_epoch_{epoch + 1}.pth'))
        torch.save(resnet.state_dict(), os.path.join('./checkpoints/', f'resnet_params_epoch_{epoch + 1}.pth'))

torch.save(net.state_dict(), './checkpoints/net_params.pth')
torch.save(resnet.state_dict(), './checkpoints/resnet_params.pth')

'''  
  recon_instance = TensorRecon()  # 创建TensorRecon实例
  reconstructed_ct = recon_instance.tensor_to_image(corrected_proj_data)
  reconstructed_ct = reconstructed_ct.to(device)
  final_ct = resnet(reconstructed_ct)
'''