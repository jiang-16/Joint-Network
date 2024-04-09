import torch
import torch.nn as nn
from pytorch_ssim import ssim  # 需要安装pytorch-ssim库

class CombinedLoss(nn.Module):
    def __init__(self, weights):
        super(CombinedLoss, self).__init__()
        self.weights = weights
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, output, target):
        mse_loss = self.mse_loss(output, target)
        mae_loss = self.mae_loss(output, target)
        ssim_loss = ssim(output, target)

        final_loss = self.weights[0] * mse_loss + self.weights[1] * mae_loss + self.weights[2] * (1 - ssim_loss)
        return final_loss

'''# 假设为损失函数的权重[0.5, 0.3, 0.2]
criterion = CombinedLoss(weights=[0.1, 0.1, 0.8])
x1 = torch.randn(1,1,512,512).cuda()
x2 = torch.randn(1,1,512,512).cuda()
print(criterion(x1,x2))
'''