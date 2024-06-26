import os
from torch.utils.data import Dataset
from utilis import *
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


class Mydataset(Dataset):
    def __init__(self,path):
        self.path = path
        self.name = os.listdir(os.path.join(path,'labelcol'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index] #xx.png
        segment_path = os.path.join(self.path,'labelcol',segment_name)
        image_path = os.path.join(self.path,'img',segment_name)
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open(image_path)
        return transform(image),transform(segment_image)


if __name__=='__main__':
    data = Mydataset('D:\\programstudy\\transformer\\Medical-Transformer-main\\data_0918')
    print(data[0][0].shape)
    print(data[0][0].shape)

