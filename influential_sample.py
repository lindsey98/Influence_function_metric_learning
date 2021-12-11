import numpy as np
import torch
import torchvision

import loss
import utils
import dataset
from loss import *
from networks import Feat_resnet50_max_n, Full_Model
from evaluation.pumap import prepare_data, get_wrong_indices
import torch.nn as nn
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torchsummary import summary
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.utils import overlay_mask
from torchvision.io.image import read_image
from evaluation.saliency_map import SmoothGrad, as_gray_image
from PIL import Image
import scipy
from torch.autograd import Variable
from influence_function import inverse_hessian_product, grad_confusion, grad_alltrain, grad_train_onecls, calc_confusion, calc_influential_func, calc_intravar, grad_intravar
import pickle
from scipy.stats import t
from utils import predict_batchwise
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

class InfluentialSample():
    def __init__(self, dataset_name, seed, loss_type, config_name,
                 measure, test_resize=True, sz_embedding=512):

        self.folder = 'models/dvi_data_{}_{}_loss{}/'.format(dataset_name, seed, loss_type)
        self.model_dir = '{}/ResNet_{}_Model'.format(self.folder, sz_embedding)
        self.measure = measure
        self.config_name = config_name
        assert self.measure in ['confusion', 'intravar']

        # load data
        self.dl_tr, self.dl_ev = prepare_data(data_name=dataset_name, config_name=self.config_name, root=self.folder, save=False, batch_size=1,
                                              test_resize=test_resize)
        self.dataset_name = dataset_name
        self.seed = seed
        self.loss_type = loss_type
        self.sz_embedding = sz_embedding
        self.model = self._load_model()
        self.criterion = self._load_criterion()
        self.train_embedding, self.train_label, self.testing_embedding, self.testing_label = self._load_data()
        pass

    def _load_model(self):
        feat = Feat_resnet50_max_n()
        emb = torch.nn.Linear(2048, self.sz_embedding)
        model = torch.nn.Sequential(feat, emb)
        model = nn.DataParallel(model)
        model.cuda()
        model.load_state_dict(torch.load('{}/Epoch_{}/{}_{}_trainval_{}_{}.pth'.format(self.model_dir, 40, self.dataset_name, self.dataset_name, self.sz_embedding, self.seed)))
        return model

    def _load_criterion(self):
        proxies = torch.load('{}/Epoch_{}/proxy.pth'.format(self.model_dir, 40), map_location='cpu')['proxies'].detach()
        if 'ProxyNCA_prob_orig' in self.loss_type:
            criterion = loss.ProxyNCA_prob_orig(nb_classes=self.dl_tr.dataset.nb_classes(), sz_embed=self.sz_embedding, scale=3)
        elif 'ProxyNCA_pfix' in self.loss_type:
            criterion = loss.ProxyNCA_pfix(nb_classes=self.dl_tr.dataset.nb_classes(), sz_embed=self.sz_embedding, scale=3)
        elif 'ProxyAnchor' in self.loss_type:
            criterion = loss.Proxy_Anchor(nb_classes=self.dl_tr.dataset.nb_classes(), sz_embed=self.sz_embedding)
        elif 'ProxyAnchor_pfix' in self.loss_type:
            criterion = loss.ProxyAnchor_pfix(nb_classes=self.dl_tr.dataset.nb_classes(), sz_embed=self.sz_embedding)
        else:
            raise NotImplementedError
        criterion.proxies.data = proxies
        criterion.cuda()
        return criterion

    def _load_data(self):
        try:
            train_embedding = torch.load('{}/Epoch_{}/training_embeddings.pth'.format(self.model_dir, 40))  # high dimensional embedding
        except FileNotFoundError:
            self.cache_embedding()
        train_embedding = torch.load('{}/Epoch_{}/training_embeddings.pth'.format(self.model_dir, 40))  # high dimensional embedding
        train_label = torch.load('{}/Epoch_{}/training_labels.pth'.format(self.model_dir, 40))
        testing_embedding = torch.load('{}/Epoch_{}/testing_embeddings.pth'.format(self.model_dir, 40))  # high dimensional embedding
        testing_label = torch.load('{}/Epoch_{}/testing_labels.pth'.format(self.model_dir, 40))

        return train_embedding, train_label, testing_embedding, testing_label

    def get_confusion_test(self, top10_wrong_classes):
        confusion_matrix = np.ones((len(top10_wrong_classes), len(self.dl_ev.dataset.classes))) * 100
        for indi, i in enumerate(top10_wrong_classes):
            for indj, j in enumerate(self.dl_ev.dataset.classes):
                if j == i: # same class
                    continue
                else:
                    feat_cls1 = self.testing_embedding[self.testing_label == i]
                    feat_cls2 = self.testing_embedding[self.testing_label == j]
                    confusion= calc_confusion(feat_cls1, feat_cls2, sqrt=True) # get t instead of t^2
                    confusion_matrix[indi][indj] = confusion.detach().cpu().item()
        return confusion_matrix

    def get_confusion_class_pairs(self):
        wrong_ind, top10_wrong_classes = get_wrong_indices(self.testing_embedding, self.testing_label, 10)
        confusion_matrix = self.get_confusion_test(top10_wrong_classes)
        np.save("Influential_data/{}_{}_top10_wrongtest_confusion.npy".format(self.dataset_name, self.loss_type), confusion_matrix)
        confusion_matrix = np.load( "Influential_data/{}_{}_top10_wrongtest_confusion.npy".format(self.dataset_name, self.loss_type))
        confusion_classes_ind = np.argsort(confusion_matrix, axis=-1)[:, 0]  # get classes that are confused with top10 wrong testing classes
        confusion_class_pairs = [(x, y) for x, y in zip(top10_wrong_classes, np.asarray(self.dl_ev.dataset.classes)[confusion_classes_ind])]
        confusion_degree = confusion_matrix[np.arange(len(confusion_matrix)), confusion_classes_ind]
        print("Top 10 confusion pairs", confusion_class_pairs)
        print("Confusion degree", confusion_degree)
        return confusion_class_pairs

    def get_intravar_test(self):
        intravar_vec = np.zeros(len(self.dl_ev.dataset.classes))
        for indj, j in enumerate(self.dl_ev.dataset.classes):
            feat_cls = self.testing_embedding[self.testing_label == j]
            intravar_vec[indj] = calc_intravar(feat_cls).detach().cpu().item()
        return intravar_vec

    def get_scatter_class(self):
        intravar_vec = self.get_intravar_test()
        np.save("Influential_data/{}_{}_intravar_test.npy".format(self.dataset_name, self.loss_type), intravar_vec)
        intravar_vec = np.load("Influential_data/{}_{}_intravar_test.npy".format(self.dataset_name, self.loss_type))

        intravar_classes_ind = np.argsort(intravar_vec)[::-1][:10]  # descending order
        scattered_classes = [x for x in np.asarray(self.dl_ev.dataset.classes)[intravar_classes_ind]]
        print("Top 10 mostly scattered testing classes", scattered_classes)
        print("Intra-class variance", intravar_vec[intravar_classes_ind])

        return scattered_classes

    def get_nearest_train_class(self, embeddings):
        dist = np.zeros(len(self.dl_tr.dataset.classes))
        for ind, cls in enumerate(self.dl_tr.dataset.classes):
            feat_cls = self.train_embedding[self.train_label == cls]
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            feat_cls = F.normalize(feat_cls, p=2, dim=-1)
            dist[ind] = torch.dot(embeddings.mean(0), feat_cls.mean(0)).item()
        nn_train_cls_ind = np.argsort(dist)[::-1] # descending
        nn_train_classes = [x for x in np.asarray(self.dl_tr.dataset.classes)[nn_train_cls_ind]]
        print("Nearest 2 training classes", nn_train_classes[:2])
        print("Furtherest 1 training class", nn_train_classes[-1])

        return nn_train_classes

    def cache_embedding(self):
        embedding, label, indices = predict_batchwise(self.model, self.dl_tr)
        torch.save(embedding, '{}/Epoch_{}/training_embeddings.pth'.format(self.model_dir, 40))
        torch.save(label, '{}/Epoch_{}/training_labels.pth'.format(self.model_dir, 40))

        testing_embedding, testing_label, testing_indices = predict_batchwise(self.model, self.dl_ev)
        torch.save(testing_embedding, '{}/Epoch_{}/testing_embeddings.pth'.format(self.model_dir, 40))
        torch.save(testing_label, '{}/Epoch_{}/testing_labels.pth'.format(self.model_dir, 40))

    def cache_grad_loss_train(self): # FIXMe: get gradient in batch
        for cls in self.dl_tr.dataset.classes:
            grad = grad_train_onecls(self.model, self.criterion, self.dl_tr, cls)
            with open("Influential_data/{}_{}_grad4traincls_{}.pkl".format(self.dataset_name, self.loss_type, cls), "wb") as fp:  # Pickling
                pickle.dump(grad, fp)

    def cache_grad_confusion_test(self, confusion_class_pairs):
        torch.cuda.empty_cache()
        for pair in confusion_class_pairs:
            v = grad_confusion(self.model, self.dl_ev, pair[0], pair[1])
            torch.save(v, "Influential_data/{}_{}_confusion_grad_test_{}_{}.pth".format(self.dataset_name, self.loss_type, pair[0], pair[1]))

    def cache_grad_intravar_test(self, scatter_class):
        for cls in scatter_class:
            v = grad_intravar(self.model, self.dl_ev, cls)
            torch.save(v, "Influential_data/{}_{}_intravar_grad_test_{}.pth".format(self.dataset_name, self.loss_type, cls))

    # def cache_ihvp(self, cls):
    #     if self.measure == 'confusion':
    #         v = torch.load("Influential_data/{}_{}_confusion_grad_test_{}_{}.pth".format(self.dataset_name, self.loss_type, cls[0], cls[1]))
    #         inverse_hvp = inverse_hessian_product(self.model, self.criterion, v, self.dl_tr)
    #         torch.save(inverse_hvp, "Influential_data/{}_{}_inverse_hvp_{}_{}.pth".format(self.dataset_name, self.loss_type, cls[0], cls[1]))
    #     else:
    #         v = torch.load("Influential_data/{}_{}_intravar_grad_test_{}.pth".format(self.dataset_name, self.loss_type, cls))
    #         inverse_hvp = inverse_hessian_product(self.model, self.criterion, v, self.dl_tr)
    #         torch.save(inverse_hvp, "Influential_data/{}_{}_inverse_hvp_{}.pth".format(self.dataset_name, self.loss_type, cls))

    def run(self, pair_idx):
        if self.measure == 'confusion':
            confusion_class_pairs = self.get_confusion_class_pairs()
            with open("Influential_data/{}_{}_grad4train.pkl".format(self.dataset_name, self.loss_type), "rb") as fp:  # Pickling
                grad4train = pickle.load(fp)
            for it, pair in enumerate(confusion_class_pairs):
                if it == pair_idx:
                    self.viz_2cls(5, self.dl_ev, self.testing_label, pair[0], pair[1]) # visualize confusion classes
                    inverse_hvp = torch.load("Influential_data/{}_{}_inverse_hvp_{}_{}.pth".format(self.dataset_name, self.loss_type, pair[0], pair[1]))
                    influence_values = calc_influential_func(inverse_hvp, grad4train)
                    influence_values = np.stack(influence_values)[:, 0]
                    influence_class = np.zeros(self.dl_tr.dataset.nb_classes())
                    for cls in range(self.dl_tr.dataset.nb_classes()):
                        influence_class[cls] = influence_values[self.train_label == cls].mean()
                    training_cls_by_influence = influence_class.argsort()[::-1] # descending
                    self.viz_2cls(5, self.dl_tr, self.train_label, training_cls_by_influence[0], training_cls_by_influence[1]) # helpful
                    self.viz_2cls(5, self.dl_tr, self.train_label, training_cls_by_influence[-1], training_cls_by_influence[-2]) # harmful

        if self.measure == 'intravar':

            scattered_classes = self.get_scatter_class()
            with open("Influential_data/{}_{}_grad4train.pkl".format(self.dataset_name, self.loss_type), "rb") as fp:  # Pickling
                grad4train = pickle.load(fp)

            for it, cls in enumerate(scattered_classes):
                if it == pair_idx:
                    self.viz_cls(5, self.dl_ev, self.testing_label, cls)
                    inverse_hvp = torch.load("Influential_data/{}_{}_inverse_hvp_{}.pth".format(self.dataset_name, self.loss_type, cls))
                    influence_values = calc_influential_func(inverse_hvp, grad4train)
                    influence_values = np.stack(influence_values)[:, 0]
                    influence_class = np.zeros(self.dl_tr.dataset.nb_classes())
                    for cls in range(self.dl_tr.dataset.nb_classes()):
                        influence_class[cls] = influence_values[self.train_label == cls].mean()
                    training_cls_by_influence = influence_class.argsort()[::-1]
                    self.viz_2cls(5, self.dl_tr, self.train_label, training_cls_by_influence[0], training_cls_by_influence[1]) # harmful
                    self.viz_2cls(5, self.dl_tr, self.train_label, training_cls_by_influence[-1], training_cls_by_influence[-2]) # helpful

    def viz_cls(self, top_bottomk, dataloader, label, cls):
        ind_cls = np.where(label.detach().cpu().numpy() == cls)[0]
        for i in range(top_bottomk):
            plt.subplot(1, top_bottomk, i + 1)
            img = read_image(dataloader.dataset.im_paths[ind_cls[i]])
            plt.imshow(to_pil_image(img))
            plt.title('Class = {}'.format(cls))
        plt.show()

    def viz_2cls(self, top_bottomk, dataloader, label, cls1, cls2):
        ind_cls1 = np.where(label.detach().cpu().numpy() == cls1)[0]
        ind_cls2 = np.where(label.detach().cpu().numpy() == cls2)[0]

        top_bottomk = min(top_bottomk, len(ind_cls1), len(ind_cls2))
        for i in range(top_bottomk):
            plt.subplot(2, top_bottomk, i + 1)
            img = read_image(dataloader.dataset.im_paths[ind_cls1[i]])
            plt.imshow(to_pil_image(img))
            plt.title('Class = {}'.format(cls1))
        for i in range(top_bottomk):
            plt.subplot(2, top_bottomk, i + 1 + top_bottomk)
            img = read_image(dataloader.dataset.im_paths[ind_cls2[i]])
            plt.imshow(to_pil_image(img))
            plt.title('Class = {}'.format(cls2))
        plt.show()


if __name__ == '__main__':

    dataset_name = 'cub'
    loss_type = 'ProxyNCA_prob_orig'
    config_name = 'cub'
    sz_embedding = 512
    seed = 0
    measure = 'confusion'

    IS = InfluentialSample(dataset_name, seed, loss_type, config_name, measure, True, sz_embedding)
    '''Step 1: Cache loss gradient to parameters for all training'''
    IS.cache_grad_loss_train()
    exit()

    '''Step 2: Cache all confusion gradient to parameters'''
    # confusion_class_pairs = IS.get_confusion_class_pairs()
    # IS.cache_grad_confusion_test(confusion_class_pairs)
    # exit()

    # scatter_class = IS.get_scatter_class()
    # IS.cache_grad_intravar_test(scatter_class)
    # exit()

    '''Step 3: Cache inverse Hessian Vector Product'''
    # for pair in confusion_class_pairs:
    #     IS.cache_ihvp(pair)
    # exit()

    # for cls in scatter_class:
    #     IS.cache_ihvp(cls)
    # exit()

    '''Step 4: Calc influence functions'''
    # IS.run(4)

    '''Step 4 (alternative): Get NN/Furtherest classes'''
    # i = 115; j = 129
    # i = 102
    # feat_cls1 = IS.testing_embedding[IS.testing_label == i]
    # feat_cls2 = IS.testing_embedding[IS.testing_label == j]
    # feat_collect = torch.cat((feat_cls1, feat_cls2))
    # feat_collect = feat_cls1
    # IS.get_nearest_train_class(feat_collect)

    '''Other: get t statistic for two specific classes'''
    # i = 7403; j = 5589
    # feat_cls1 = IS.testing_embedding[IS.testing_label == i]
    # feat_cls2 = IS.testing_embedding[IS.testing_label == j]
    # confusion = calc_confusion(feat_cls1, feat_cls2, sqrt=True)  # get t instead of t^2
    # print(confusion.item())

    '''Other: get intra-class variance for a specific class'''
    # i = 102
    # feat_cls = IS.testing_embedding[IS.testing_label == i]
    # intra_var = calc_intravar(feat_cls)
    # print(intra_var.item())

    '''Other: visualize two specific classes'''
    # IS.viz_2cls(5, IS.dl_ev, IS.testing_label, 21626, 15606 )
    # ()




