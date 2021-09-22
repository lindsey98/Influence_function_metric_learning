import torchvision.datasets.folder

from .base import *
import h5py
import torch
from typing import List
import math
import shutil
import json

class Logo2k(BaseDatasetMod):
    def __init__(self, root, source, classes, transform=None):
        BaseDatasetMod.__init__(self, root, source, classes, transform)
        index = 0
        dataloader = torchvision.datasets.ImageFolder(root=root)
        dataloader.classes = [str(x) for x in sorted(int(d) for d in os.listdir(dataloader.root) if os.path.isdir(os.path.join(dataloader.root, d)))]
        dataloader.class_to_idx = {dataloader.classes[i]: i for i in range(len(dataloader.classes))}
        dataloader.samples = dataloader.make_dataset(dataloader.root, dataloader.class_to_idx, torchvision.datasets.folder.IMG_EXTENSIONS, None)
        dataloader.imgs = dataloader.samples
        for i in dataloader.imgs:
            # i[1]: label, i[0]: root
            y = i[1]
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if y in self.classes and fn[:2] != '._':
                self.ys += [y]
                self.I += [index]
                # self.im_paths.append(os.path.join(root, i[0]))
                self.im_paths.append(i[0])
                index += 1


class Logo2k_super(BaseDatasetMod):
    def __init__(self, root, source, classes, transform=None):
        BaseDatasetMod.__init__(self, root, source, classes, transform)
        index = 0
        dataloader = torchvision.datasets.ImageFolder(root=root)
        dataloader.classes = [str(x) for x in sorted(int(d) for d in os.listdir(dataloader.root) if os.path.isdir(os.path.join(dataloader.root, d)))]
        dataloader.class_to_idx = {dataloader.classes[i]: i for i in range(len(dataloader.classes))}
        dataloader.samples = dataloader.make_dataset(dataloader.root, dataloader.class_to_idx, torchvision.datasets.folder.IMG_EXTENSIONS, None)
        dataloader.imgs = dataloader.samples

        for i in dataloader.imgs:
            # i[1]: label, i[0]: root
            y = i[1]
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if y in self.classes and fn[:2] != '._':
                self.ys += [y]
                self.I += [index]
                # self.im_paths.append(os.path.join(root, i[0]))
                self.im_paths.append(i[0])
                index += 1
        pass

class Logo2k_class(BaseDatasetMod):
    def __init__(self, root, source, classes, transform=None, mode='train'):
        BaseDatasetMod.__init__(self, root, source, classes, transform)
        index = 0

        for i in torchvision.datasets.ImageFolder(root=os.path.join(root, 'images')).imgs:
            # i[1]: label, i[0]: root
            y = i[1]
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if y in self.classes and fn[:2] != '._':
                self.ys += [y]
                self.I += [index]
                # self.im_paths.append(os.path.join(root, i[0]))
                self.im_paths.append(i[0])
                index += 1

        cut_off = int(len(self.ys) * 0.5)
        torch.manual_seed(1)
        rand_list = torch.randperm(len(self.ys)).tolist()

        ys = []
        I = []
        paths = []
        if mode == 'train':
            for ix in range(len(self.ys)):
                if ix < cut_off:
                    ys.append(self.ys[rand_list[ix]])
                    I.append(self.I[rand_list[ix]])
                    paths.append(self.im_paths[rand_list[ix]])
        else:
            for ix in range(len(self.ys)):
                if ix >= cut_off:
                    ys.append(self.ys[rand_list[ix]])
                    I.append(self.I[rand_list[ix]])
                    paths.append(self.im_paths[rand_list[ix]])

        self.ys = ys
        self.I = I
        self.im_paths = paths


