from __future__ import print_function
from __future__ import division

import torchvision
from torchvision import transforms
import PIL.Image
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# from torch._six import int_classes as _int_classes
_int_classes = int
import numpy as np
import numbers
import torch.nn.functional as F
import logging
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
import scipy
from scipy.spatial import distance


def std_per_channel(images):
    images = torch.stack(images, dim = 0)
    return images.view(3, -1).std(dim = 1)


def mean_per_channel(images):
    images = torch.stack(images, dim = 0)
    return images.view(3, -1).mean(dim = 1)


class Identity(): # used for skipping transforms
    def __call__(self, im):
        return im

class RGBAToRGB():
    def __call__(self, im):
        im = im.convert('RGB')
        return im

class RGBToBGR():
    def __call__(self, im):
        assert im.mode == 'RGB'
        r, g, b = [im.getchannel(i) for i in range(3)]
        # RGB mode also for BGR, `3x8-bit pixels, true color`, see PIL doc
        im = PIL.Image.merge('RGB', [b, g, r])
        return im


class ScaleIntensities():
    def __init__(self, in_range, out_range):
        """ Scales intensities. For example [-1, 1] -> [0, 255]."""
        self.in_range = in_range
        self.out_range = out_range

    def __oldcall__(self, tensor):
        tensor.mul_(255)
        return tensor

    def __call__(self, tensor):
        tensor = (
            tensor - self.in_range[0]
        ) / (
            self.in_range[1] - self.in_range[0]
        ) * (
            self.out_range[1] - self.out_range[0]
        ) + self.out_range[0]
        return tensor


def make_transform(sz_resize = 256, sz_crop = 227,
                   mean = [104, 117, 128], std = [1, 1, 1],
                   rgb_to_bgr = True, is_train = True,
                   intensity_scale = None, rotate = 0):
    return transforms.Compose([
        RGBAToRGB(),
        RGBToBGR() if rgb_to_bgr else Identity(),
        transforms.RandomRotation(rotate) if is_train and (not isinstance(rotate, numbers.Number)) else Identity(),
        transforms.RandomResizedCrop(sz_crop) if is_train else Identity(),
        transforms.Resize(sz_resize) if not is_train else Identity(),
        transforms.CenterCrop(sz_crop) if not is_train else Identity(),
        transforms.RandomHorizontalFlip() if is_train else Identity(),
        transforms.ToTensor(),
        ScaleIntensities(
            *intensity_scale) if intensity_scale is not None else Identity(),
        transforms.Normalize(
            mean=mean,
            std=std,
        )
    ])

class BatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, batch_size, drop_last, dataset, sel_class):
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.dataset = dataset

        self.dataset.pop_class_list()
        self.dataset.sel_class = sel_class
        self.dataset.resel_random_classes()

    def __iter__(self):
        batch = []
        for idx in range(len(self.dataset)):
            rand_class = self.dataset.random_classes[idx % self.dataset.sel_class]
            class_list = self.dataset.class_list[rand_class]
            idx = self.dataset.class_list[rand_class][torch.randperm(len(class_list))[0]]
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                self.dataset.resel_random_classes()
        if len(batch) > 0 and not self.drop_last:
            yield batch
    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

class RandomBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, labels, batch_size, drop_last, sel_class, nb_gradcum=1):
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.nb_gradcum = nb_gradcum

        self.labels = labels
        self.class_list = []
        for ix in range(len(set(labels))): self.class_list.append([])
        for ix in range(len(labels)):
            self.class_list[labels[ix]].append(ix)
        self.sel_class = sel_class
        self.random_classes = torch.randperm(len(self.class_list))[:self.sel_class]


    def __iter__(self):
        batch = []
        bc = 0
        #for idx in range(len(self.dataset)):
        for idx in range(len(self.labels)):
            #rand_class = self.dataset.random_classes[idx % self.dataset.sel_class]
            rand_class = self.random_classes[torch.randperm(len(self.random_classes))[0]]
            class_list = self.class_list[rand_class]
            idx = self.class_list[rand_class][torch.randperm(len(class_list))[0]]
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                bc += 1
                #self.dataset.resel_random_classes()
            if bc == self.nb_gradcum:
                bc = 0
                #self.dataset.resel_random_classes()
                self.random_classes = torch.randperm(len(self.class_list))[:self.sel_class]
        if len(batch) > 0 and not self.drop_last:
            yield batch
    def __len__(self):
        if self.drop_last:
            return len(self.labels) // self.batch_size
        else:
            return (len(self.labels) + self.batch_size - 1) // self.batch_size

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:\
                               self.used_label_indices_count[class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


class BalancedBatchExcludeSampler(BatchSampler):
    """
    BatchSampler -samples n_classes and within these classes samples n_samples, but exclude some of the indices
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples, exclude_ind):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.exclude_ind = exclude_ind
        self._remove_exclude()

        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels) - len(exclude_ind)
        self.batch_size = self.n_samples * self.n_classes

    def _remove_exclude(self):
        # remove excluded_inds
        for l in self.labels_set:
            compare = self.label_to_indices[l][:, None] == self.exclude_ind # (N, N_exclude)
            isexclude = compare.sum(-1) # (N,)
            self.label_to_indices[l] = np.delete(self.label_to_indices[l], isexclude == True)

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset: # exceed number of data we have
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False) # randomlly choose n classes

            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:\
                               self.used_label_indices_count[class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples

                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0

            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


class ClsDistSampler(torch.utils.data.sampler.Sampler):
    """
    Plugs into PyTorch Batchsampler Package.
    """
    def __init__(self, labels, n_classes, n_samples):

        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def create_storage(self, dataloader, model): # this dataloader should be no shuffled version

        from utils import predict_batchwise
        X, T, *_ = predict_batchwise(model, dataloader)

        # similarity matrix
        cls_sim_matrix = torch.zeros((len(self.labels_set), len(self.labels_set)))
        for i in self.labels_set:
            for j in self.labels_set:
                if j > i:
                    indices_i = self.label_to_indices[i]
                    indices_j = self.label_to_indices[j]
                    Xi, Xj = X[indices_i], X[indices_j]
                    sim = self.inner_prod(Xi, Xj) # get sample pairwise similarity
                    cls_sim_matrix[int(i), int(j)] = (sim.item() + 1.)
                    cls_sim_matrix[int(j), int(i)] = (sim.item() + 1.)

        normalization_factor = torch.sum(cls_sim_matrix, dim=1, keepdim=True)
        cls_sim_matrix = cls_sim_matrix / normalization_factor  # normalize to be between 0 and 1
        self.storage = cls_sim_matrix
        logging.info('Reinitialize Class Sampler')

    def inner_prod(self, X1, X2):
        X1, X2 = F.normalize(X1, dim=-1), F.normalize(X2, dim=-1)
        sim = torch.matmul(X1, X2.T)
        return sim.mean()


    def diverse_class_sampling(self):
        chosen_cls = np.random.choice(self.labels_set, 1, replace=False)[0] # first class is chosen at random

        prob = self.storage[int(chosen_cls), :].numpy()
        prob = ((1. - prob) * (prob != 0))  # sample far away class to have more diverse selection
        rest_classes = np.random.choice(self.labels_set, self.n_classes-1,
                                        p=(prob + 1e-8) / (prob.sum() + 1e-8), replace=False)
        classes = [chosen_cls] + rest_classes.tolist()

        return classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            # random sample a class and the other classes
            classes = self.diverse_class_sampling()

            indices = []
            for cls in classes:
                indices.extend(self.label_to_indices[cls][self.used_label_indices_count[cls]:\
                                                            (self.used_label_indices_count[cls] + self.n_samples)])
                self.used_label_indices_count[cls] += self.n_samples

                if self.used_label_indices_count[cls] + self.n_samples > len(self.label_to_indices[cls]):
                    np.random.shuffle(self.label_to_indices[cls])
                    self.used_label_indices_count[cls] = 0

            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

class ClsCohSampler(torch.utils.data.sampler.Sampler):
    """
    Sample class by compression degree
    """

    def __init__(self, labels, n_classes, n_samples):

        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def create_storage(self, dataloader, model):  # this dataloader should be no shuffled version

        from utils import predict_batchwise
        X, T, *_ = predict_batchwise(model, dataloader)

        # similarity matrix
        self.intra_inter_ratio = self.get_intra_inter_dist(X, T)
        self.storage = self.get_class_svd(X, T)
        logging.info('Reinitialize Class Sampler')

    def get_class_svd(self, X, T):
        X = F.normalize(X, p=2, dim=-1)
        rho_values = []
        for cls in self.labels_set:
            indices = T == cls
            X_cls = X[indices, :]  # class-specific embedding
            u, s, v = torch.linalg.svd(X_cls)  # compute singular value, lower value implies lower data variance
            s = s[1:].detach().cpu().numpy()  # remove first singular value cause it is over-dominant
            # TODO: use the definition in "Revisiting Training Strategies and Generalization Performance in Deep Metric"
            s_norm = s / s.sum()
            uniform = np.ones(len(s)) / (len(s))
            kl = scipy.stats.entropy(uniform, s_norm)
            rho_values.append(kl)
        return rho_values # (C,)

    def get_intra_inter_dist(self, X, T):
        X = F.normalize(X, p=2, dim=-1)
        X = X.detach().cpu().numpy()
        dist_mat = np.zeros((len(self.labels_set), len(self.labels_set)))

        # Get class-specific embedding
        X_arrange_byT = []
        for cls in self.labels_set:
            indices = T == cls
            X_cls = X[indices, :]
            X_arrange_byT.append(X_cls)

        # O(C^2) to calculate inter, intra distance
        for i in range(len(self.labels_set)):
            for j in range(i, len(self.labels_set)):
                pairwise_dists = distance.cdist(X_arrange_byT[i], X_arrange_byT[j], 'cosine')
                avg_pairwise_dist = np.sum(pairwise_dists) / (np.prod(pairwise_dists.shape) - len(pairwise_dists.diagonal())) # take mean (ignore diagonal)
                dist_mat[i, j] = dist_mat[j, i] = avg_pairwise_dist

        ratio_mat = dist_mat / (dist_mat.diagonal()[:, np.newaxis]+1e-8) # (C, C) inter/intra ratio matrix
        ratio_mat = 1./(ratio_mat + 1e-8) # (C, C) intra/inter distance ratio
        # if this ratio is sufficiently low, you dont want to optimize any more
        ratio_mat = ratio_mat.mean(-1) # (C,)
        return ratio_mat

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            # sample large variance classes
            classes_prob = -np.asarray(self.storage) # lower rho -> more directions with significant variance -> choose
            classes_prob = classes_prob / classes_prob.sum()
            classes = np.random.choice(self.labels_set, self.n_classes,
                             p=classes_prob, replace=False)
            # sample large intra/inter ratio classes
            # classes_prob = np.asarray(self.intra_inter_ratio)  # larger intra/inter ratio -> densely populated embedding space, low compression degree, directions with significant variance -> choose
            # classes_prob = classes_prob / classes_prob.sum()
            # classes = np.random.choice(self.labels_set, self.n_classes,
            #                            p=classes_prob, replace=False)

            indices = []
            for cls in classes:
                indices.extend(self.label_to_indices[cls][self.used_label_indices_count[cls]: \
                                                          (self.used_label_indices_count[cls] + self.n_samples)])
                self.used_label_indices_count[cls] += self.n_samples

                if self.used_label_indices_count[cls] + self.n_samples > len(self.label_to_indices[cls]):
                    np.random.shuffle(self.label_to_indices[cls])
                    self.used_label_indices_count[cls] = 0

            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

class ClsSubsetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes
        self.storage = None

    def create_storage(self, test_loader, train_loader, model):  # this dataloader should be test/val loader

        from utils import get_wrong_indices, predict_batchwise, pairwise_distance
        X, T, *_ = predict_batchwise(model, test_loader)
        _, top15wrongclasses = get_wrong_indices(X, T)
        top15wrongclasses = torch.from_numpy(top15wrongclasses)
        wrongcls_emb = X[torch.isin(T, top15wrongclasses)]

        # find neighboring training classes
        X_tr, T_tr, *_ = predict_batchwise(model, train_loader)
        _, IP = pairwise_distance(
            torch.cat([X_tr, wrongcls_emb]),
            squared=True
        )
        IP = IP[:len(X_tr), len(X_tr):] # affinity to wrong classes' embeddings (N_tr, N')

        M = torch.zeros((train_loader.dataset.nb_classes(), len(X_tr))) # (C_tr, N_tr)
        M[T_tr, torch.arange(len(X_tr))] = 1
        M = torch.nn.functional.normalize(M, p=1, dim=1) # average matrix average over all training samples for each class
        similarity = torch.mm(M, IP) # (C_tr, N')
        similarity = similarity.max(-1)[0] # (C_tr)
        self.storage = similarity
        logging.info('Reinitialize Class Sampler by finding probably wrong region')

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            if isinstance(self.storage, torch.Tensor):
                # top classes that lie in the mostly likely wrong regions
                classes = torch.argsort(self.storage, descending=True)[:self.n_classes].numpy()
            else:
                classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for cls in classes:
                indices.extend(self.label_to_indices[cls][self.used_label_indices_count[cls]: \
                                                          (self.used_label_indices_count[cls] + self.n_samples)])
                self.used_label_indices_count[cls] += self.n_samples

                if self.used_label_indices_count[cls] + self.n_samples > len(self.label_to_indices[cls]):
                    np.random.shuffle(self.label_to_indices[cls])
                    self.used_label_indices_count[cls] = 0

            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
