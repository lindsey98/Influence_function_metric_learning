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
from influence_function import *
import pickle
from scipy.stats import t
from utils import predict_batchwise
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

class InfluentialSample():
    def __init__(self, dataset_name, seed, loss_type, config_name,
                 measure, test_resize=True, sz_embedding=512, epoch=40):

        self.folder = 'dvi_data_{}_{}_loss{}/'.format(dataset_name, seed, loss_type)
        self.model_dir = '{}/ResNet_{}_Model'.format(self.folder, sz_embedding)
        self.measure = measure
        self.config_name = config_name
        self.epoch = epoch
        assert self.measure in ['confusion', 'intravar']

        # load data
        self.dl_tr, self.dl_ev = prepare_data(data_name=dataset_name, config_name=self.config_name, root=self.folder,
                                              save=False, batch_size=1,
                                              test_resize=test_resize)
        self.dataset_name = dataset_name
        self.seed = seed
        self.loss_type = loss_type
        self.sz_embedding = sz_embedding
        self.model = self._load_model()
        self.criterion = self._load_criterion()
        self.train_embedding, self.train_label, self.testing_embedding, self.testing_label = self._load_data()

    def _load_model(self):
        feat = Feat_resnet50_max_n()
        emb = torch.nn.Linear(2048, self.sz_embedding)
        model = torch.nn.Sequential(feat, emb)
        model = nn.DataParallel(model)
        model.cuda()
        model.load_state_dict(torch.load('{}/Epoch_{}/{}_{}_trainval_{}_{}.pth'.format(self.model_dir, self.epoch, self.dataset_name, self.dataset_name, self.sz_embedding, self.seed)))
        return model

    def _load_criterion(self):
        proxies = torch.load('{}/Epoch_{}/proxy.pth'.format(self.model_dir, self.epoch), map_location='cpu')['proxies'].detach()
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

    def cache_embedding(self):
        embedding, label, indices = predict_batchwise(self.model, self.dl_tr)
        torch.save(embedding, '{}/Epoch_{}/training_embeddings.pth'.format(self.model_dir, self.epoch))
        torch.save(label, '{}/Epoch_{}/training_labels.pth'.format(self.model_dir, self.epoch))

        testing_embedding, testing_label, testing_indices = predict_batchwise(self.model, self.dl_ev)
        torch.save(testing_embedding, '{}/Epoch_{}/testing_embeddings.pth'.format(self.model_dir, self.epoch))
        torch.save(testing_label, '{}/Epoch_{}/testing_labels.pth'.format(self.model_dir, self.epoch))

    def _load_data(self):
        try:
            train_embedding = torch.load('{}/Epoch_{}/training_embeddings.pth'.format(self.model_dir, self.epoch))  # high dimensional embedding
        except FileNotFoundError:
            self.cache_embedding()
        train_embedding = torch.load('{}/Epoch_{}/training_embeddings.pth'.format(self.model_dir, self.epoch))  # high dimensional embedding
        train_label = torch.load('{}/Epoch_{}/training_labels.pth'.format(self.model_dir, self.epoch))
        testing_embedding = torch.load('{}/Epoch_{}/testing_embeddings.pth'.format(self.model_dir, self.epoch))  # high dimensional embedding
        testing_label = torch.load('{}/Epoch_{}/testing_labels.pth'.format(self.model_dir, self.epoch))

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
        print("Nearest 5 training classes", nn_train_classes[:5])
        print("Furtherest 5 training class", nn_train_classes[-5:])

        return nn_train_classes

    def cache_grad_loss_train(self, theta, theta_hat, pair_idx):
        l_prev, l_cur = loss_change_train(self.model, self.criterion, self.dl_tr, theta, theta_hat)
        if self.measure == 'confusion':
            for cls in self.dl_tr.dataset.classes:
                l_prev_cls = l_prev[IS.train_label == cls]
                l_cur_cls = l_cur[IS.train_label == cls]
                grad_loss = {'l_prev': l_prev_cls, 'l_cur': l_cur_cls}
                with open("Influential_data/{}_{}_confusion_grad4traincls_{}_testpair{}.pkl".format(self.dataset_name, self.loss_type, cls, pair_idx), "wb") as fp:  # Pickling
                    pickle.dump(grad_loss, fp)

        elif self.measure == 'intravar':
            for cls in self.dl_tr.dataset.classes:
                l_prev_cls = l_prev[IS.train_label == cls]
                l_cur_cls = l_cur[IS.train_label == cls]
                grad_loss = {'l_prev': l_prev_cls, 'l_cur': l_cur_cls}
                with open("Influential_data/{}_{}_intravar_grad4traincls_{}_testcls{}.pkl".format(self.dataset_name, self.loss_type, cls, pair_idx), "wb") as fp:  # Pickling
                    pickle.dump(grad_loss, fp)

    def theta_grad_descent(self, classes, n=50, lr=0.001):
        theta_orig = self.model.module[-1].weight.data
        if self.measure == 'confusion':
            torch.cuda.empty_cache()
            for pair in classes:
                confusion_degree_orig, _ = grad_confusion(self.model, self.dl_ev, pair[0], pair[1])
                confusion_degree_orig = confusion_degree_orig.detach()
                self.model.module[-1].weight.data = theta_orig
                theta = theta_orig.clone()
                for _ in range(n):
                    confusion_degree, v = grad_confusion(self.model, self.dl_ev, pair[0], pair[1]) # dt2/dtheta
                    confusion_degree = confusion_degree.detach()
                    v = v[0].detach()
                    if confusion_degree.item() - confusion_degree_orig.item() >= 0.5: # FIXME: threshold is 3.
                        break
                    theta_new = theta + lr * v # gradient ascent
                    theta = theta_new
                    self.model.module[-1].weight.data = theta

                theta_dict = {'theta': theta_orig, 'theta_hat': theta}
                torch.save(theta_dict, "Influential_data/{}_{}_confusion_theta_test_{}_{}.pth".format(self.dataset_name, self.loss_type,
                                                                                                      pair[0], pair[1])) # FIXME
        if self.measure == 'intravar':
            torch.cuda.empty_cache()
            for cls in classes:
                self.model.module[-1].weight.data = theta_orig
                intravar_orig, _ = grad_intravar(self.model, self.dl_ev, cls)  # dvar/dtheta
                intravar_orig = intravar_orig.detach()
                theta = theta_orig.clone()
                for _ in range(n):
                    intravar, v = grad_intravar(self.model, self.dl_ev, cls) # dvar/dtheta
                    intravar = intravar.detach()
                    v = v[0].detach()
                    if intravar_orig.item() - intravar.item() >= 50.: # FIXME:
                        break
                    theta_new = theta - lr * v # gradient descent
                    theta = theta_new
                    self.model.module[-1].weight.data = theta

                theta_dict = {'theta': theta_orig, 'theta_hat': theta}
                torch.save(theta_dict, "Influential_data/{}_{}_intravar_theta_test_{}.pth".format(self.dataset_name, self.loss_type,
                                                                                                      cls))
        self.model.module[-1].weight.data = theta_orig
        pass

    def run(self, pair_idx):
        if self.measure == 'confusion':
            confusion_class_pairs = self.get_confusion_class_pairs()
            pair = confusion_class_pairs[pair_idx]
            self.viz_2cls(5, self.dl_ev, self.testing_label, pair[0], pair[1])  # visualize confusion classes

            grad4train = []
            for cls in self.dl_tr.dataset.classes:
                with open("Influential_data/{}_{}_confusion_grad4traincls_{}_testpair{}.pkl".format(self.dataset_name, self.loss_type, cls, pair_idx), "rb") as fp:  # Pickling
                    grad4train_cls = pickle.load(fp)
                grad4train.append(grad4train_cls)

        elif self.measure == 'intravar':

            scattered_classes = self.get_scatter_class()
            scatter_cls = scattered_classes[pair_idx]
            self.viz_cls(5, self.dl_ev, self.testing_label, scatter_cls)

            grad4train = []
            for cls in self.dl_tr.dataset.classes:
                with open("Influential_data/{}_{}_intravar_grad4traincls_{}_testcls{}.pkl".format(self.dataset_name, self.loss_type, cls, pair_idx), "rb") as fp:  # Pickling
                    grad4train_cls = pickle.load(fp)
                grad4train.append(grad4train_cls)

        else:
            raise NotImplementedError

        influence_values = calc_influential_func(grad4train)
        influence_values = np.asarray(influence_values)
        training_cls_by_influence = np.asarray(self.dl_tr.dataset.classes)[influence_values.argsort()[::-1]]  # decending
        print(training_cls_by_influence[:5])  # helpful
        print(training_cls_by_influence[-5:])  # harmful
        self.viz_2cls(5, self.dl_tr, self.train_label, training_cls_by_influence[0],
                      training_cls_by_influence[1])  # helpful
        self.viz_2cls(5, self.dl_tr, self.train_label, training_cls_by_influence[-1],
                      training_cls_by_influence[-2])  # harmful

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
    loss_type = 'ProxyNCA_pfix_confusion_143_140'
    config_name = 'cub'
    sz_embedding = 512
    seed = 4
    measure = 'confusion'
    # epoch = 40
    epoch = 10

    IS = InfluentialSample(dataset_name, seed, loss_type, config_name, measure, True, sz_embedding, epoch)

    '''Step 1: Cache all confusion gradient to parameters'''
    # confusion_class_pairs = IS.get_confusion_class_pairs()
    # IS.theta_grad_descent(confusion_class_pairs)
    # exit()

    # scatter_classes = IS.get_scatter_class()
    # IS.theta_grad_descent(scatter_classes)
    # exit()

    '''Step 2: Cache training class loss changes'''
    # pair_idx = 3
    # i = confusion_class_pairs[pair_idx][0]; j = confusion_class_pairs[pair_idx][1]
    # theta_dict = torch.load("Influential_data/{}_{}_confusion_theta_test_{}_{}.pth".format(IS.dataset_name, IS.loss_type, i, j))
    # theta = theta_dict['theta']
    # theta_hat = theta_dict['theta_hat']
    # IS.cache_grad_loss_train(theta, theta_hat, pair_idx)
    # exit()

    # i = 136
    # theta_dict = torch.load("Influential_data/{}_{}_intravar_theta_test_{}.pth".format(IS.dataset_name, IS.loss_type, i))
    # theta = theta_dict['theta']
    # theta_hat = theta_dict['theta_hat']
    # IS.cache_grad_loss_train(theta, theta_hat)
    # exit()

    '''Step 3: Calc influence functions'''
    # IS.run(3)
    # exit()

    '''Step 4 (alternative): Get NN/Furtherest classes'''
    # i = 143; j = 145
    # i = 102
    # feat_cls1 = IS.testing_embedding[IS.testing_label == i]
    # feat_cls2 = IS.testing_embedding[IS.testing_label == j]
    # feat_collect = torch.cat((feat_cls1, feat_cls2))
    # feat_collect = feat_cls1
    # IS.get_nearest_train_class(feat_collect)
    # exit()

    '''Other: get t statistic for two specific classes'''
    i = 143; j = 140
    feat_cls1 = IS.testing_embedding[IS.testing_label == i]
    feat_cls2 = IS.testing_embedding[IS.testing_label == j]
    confusion = calc_confusion(feat_cls1, feat_cls2, sqrt=True)  # get t instead of t^2
    print(confusion.item())

    '''Other: get intra-class variance for a specific class'''
    # i = 136
    # feat_cls = IS.testing_embedding[IS.testing_label == i]
    # intra_var = calc_intravar(feat_cls)
    # print(intra_var.item())

    '''Other: visualize two specific classes'''
    # IS.viz_2cls(5, IS.dl_ev, IS.testing_label, 21626, 15606 )
    # ()




