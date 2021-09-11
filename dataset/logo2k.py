from .base import *
import h5py
import torch

class Logo2k(BaseDatasetMod):
    def __init__(self, root, source, classes, transform=None):
        BaseDatasetMod.__init__(self, root, source, classes, transform)
        index = 0
        for i in torchvision.datasets.ImageFolder(root=root).imgs:
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
