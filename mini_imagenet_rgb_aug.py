import os.path as osp
from PIL import Image
import torch
import cv2
from torch.utils.data import Dataset
from torchvision import transforms


ROOT_PATH = '/home/shenyq/data/mini/rgb/'
ROOT_PATH2 = '/home/shenyq/data/mini/rgb/'
root_mask = '/home/shenyq/data/mini/mask_deeplabv3/'
root_bg = '/home/shenyq/data/mini/bg'


class MiniImageNet(Dataset):

    def __init__(self, setname):
        csv_path = osp.join(ROOT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        mask = []
        bg = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(ROOT_PATH2, 'images', name)
            path_mask = osp.join(root_mask, 'images', name)
            path_bg =osp.join(root_bg, 'images', name)
            # path = osp.join(ROOT_PATH, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            mask.append(path_mask)
            bg.append(path_bg)
            label.append(lb)

        self.data = data
        self.mask = mask
        self.bg = bg
        self.label = label

        # self.transform = transforms.Compose([
        #     transforms.Resize(84),
        #     transforms.CenterCrop(84),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        # ])
        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, path_mask,path_bg,label = self.data[i],self.mask[i],self.bg[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        mask = cv2.imread(path_mask)
        mask = cv2.resize(mask,(84,84))
        mask = torch.from_numpy(mask).permute(2,0,1)

        bg = self.transform(Image.open(path_bg).convert('RGB'))
        return image,mask,bg, label

