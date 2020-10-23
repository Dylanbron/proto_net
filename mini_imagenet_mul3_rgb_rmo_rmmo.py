import os.path as osp
from PIL import Image
import torch
import cv2
from torch.utils.data import Dataset
from torchvision import transforms


ROOT_PATH = '/home/shenyq/data/mini/rgb/'
ROOT_PATH2 = '/home/shenyq/data/mini/rgb/'
root_img_mask = '/home/shenyq/data/mini/result_u2_select/img_modify/'
root_img_mask_mo = '/home/shenyq/data/mini/result_u2_select/img_mask_u2_modify_s/'


class MiniImageNet(Dataset):

    def __init__(self, setname):
        csv_path = osp.join(ROOT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        img_mask = []
        img_mask_mo = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(ROOT_PATH2, 'images', name)
            path_img_mask = osp.join(root_img_mask, 'images', name)
            path_img_mask_mo =osp.join(root_img_mask_mo, 'images', name)
            # path = osp.join(ROOT_PATH, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            img_mask.append(path_img_mask)
            img_mask_mo.append(path_img_mask_mo)
            label.append(lb)

        self.data = data
        self.img_mask = img_mask
        self.img_mask_mo = img_mask_mo
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
        path, path_img_mask,path_img_mask_mo,label = self.data[i],self.img_mask[i],self.img_mask_mo[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        img_mask = self.transform(Image.open(path_img_mask).convert('RGB'))
        img_mask_mo = self.transform(Image.open(path_img_mask_mo).convert('RGB'))
        return image,img_mask,img_mask_mo, label

