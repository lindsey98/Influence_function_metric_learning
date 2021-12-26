import numpy as np
import torch
import torchvision

import loss
from loss import *
from networks import Feat_resnet50_max_n, Full_Model
from evaluation.pumap import prepare_data, get_wrong_indices
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.io.image import read_image
from Influence_function.influence_function import *
import pickle
from utils import predict_batchwise
from collections import OrderedDict
import scipy.stats
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

class InfluentialSample():
    def __init__(self, dataset_name, seed, loss_type, config_name,
                 measure, test_crop=True, sz_embedding=512, epoch=40):

        self.folder = 'models/dvi_data_{}_{}_loss{}/'.format(dataset_name, seed, loss_type)
        self.model_dir = '{}/ResNet_{}_Model'.format(self.folder, sz_embedding)
        self.measure = measure
        self.config_name = config_name
        self.epoch = epoch
        assert self.measure in ['confusion', 'intravar']

        # load data
        self.dl_tr, self.dl_ev = prepare_data(data_name=dataset_name,
                                              config_name=self.config_name,
                                              root=self.folder,
                                              save=False, batch_size=1,
                                              test_crop=test_crop)
        self.dataset_name = dataset_name
        self.seed = seed
        self.loss_type = loss_type
        self.sz_embedding = sz_embedding
        self.model = self._load_model()
        self.criterion = self._load_criterion()
        self.train_embedding, self.train_label, self.testing_embedding, self.testing_label = self._load_data()

    def _load_model(self, multi_gpu=True):
        feat = Feat_resnet50_max_n()
        emb = torch.nn.Linear(2048, self.sz_embedding)
        model = torch.nn.Sequential(feat, emb)
        weights = torch.load(
            '{}/Epoch_{}/{}_{}_trainval_{}_{}.pth'.format(self.model_dir, self.epoch, self.dataset_name,
                                                          self.dataset_name, self.sz_embedding, self.seed))
        if multi_gpu:
            model = nn.DataParallel(model)
        model.cuda()
        if multi_gpu:
            model.load_state_dict(weights)
        else:
            weights_detach = OrderedDict()
            for k, v in weights.items():
                weights_detach[k.split('module.')[1]] = v
            model.load_state_dict(weights_detach)
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
        embedding, label, _ = predict_batchwise(self.model, self.dl_tr)
        torch.save(embedding, '{}/Epoch_{}/training_embeddings.pth'.format(self.model_dir, self.epoch))
        torch.save(label, '{}/Epoch_{}/training_labels.pth'.format(self.model_dir, self.epoch))

        testing_embedding, testing_label, _ = predict_batchwise(self.model, self.dl_ev)
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
        confusion_matrix = np.zeros((len(top10_wrong_classes), len(self.dl_ev.dataset.classes)))
        for indi, i in enumerate(top10_wrong_classes):
            for indj, j in enumerate(self.dl_ev.dataset.classes):
                if j == i: # same class
                    continue
                else:
                    feat_cls1 = self.testing_embedding[self.testing_label == i]
                    feat_cls2 = self.testing_embedding[self.testing_label == j]
                    confusion = calc_inter_dist(feat_cls1, feat_cls2)
                    confusion_matrix[indi][indj] = confusion.detach().cpu().item()
        return confusion_matrix

    def get_wrong_freq_matrix(self, top10_wrong_classes, wrong_labels, wrong_preds):

        wrong_freq_matrix = np.zeros((len(top10_wrong_classes), len(self.dl_ev.dataset.classes)))
        for indi, i in enumerate(top10_wrong_classes):
            ct_wrong = np.sum(wrong_labels.detach().cpu().numpy() == i)
            for indj, j in enumerate(self.dl_ev.dataset.classes):
                if i == j:
                    continue
                ct_wrong_as_clsj = np.sum((wrong_preds.squeeze().detach().cpu().numpy() == j) & (wrong_labels.detach().cpu().numpy() == i))
                wrong_freq_matrix[indi][indj] = - ct_wrong_as_clsj / ct_wrong # negative for later sorting purpose

        return wrong_freq_matrix

    def get_confusion_class_pairs(self):
        _, top10_wrong_classes, wrong_labels, wrong_preds = get_wrong_indices(self.testing_embedding, self.testing_label, 10)
        confusion_degree_matrix = self.get_confusion_test(top10_wrong_classes) # (10, C)

        wrong_freq_matrix = self.get_wrong_freq_matrix(top10_wrong_classes, wrong_labels, wrong_preds)
        confusion_classes_ind = np.argsort(wrong_freq_matrix, axis=-1)[:, :10]  # get 10 classes that are neighboring with top 10 wrong testing classes, 10 x 10
        confusion_class_pairs = [(top10_wrong_classes[i], np.asarray(self.dl_ev.dataset.classes)[confusion_classes_ind[i][j]]) \
                                 for i in range(confusion_classes_ind.shape[0]) for j in range(confusion_classes_ind.shape[1])]

        wrong_degree = np.zeros((confusion_classes_ind.shape[0], confusion_classes_ind.shape[1]))
        for i in range(confusion_classes_ind.shape[0]):
            for j in range(confusion_classes_ind.shape[1]):
                wrong_degree[i][j] = -wrong_freq_matrix[i, confusion_classes_ind[i][j]]
        wrong_degree_entropy = np.nansum(-wrong_degree * np.log2(wrong_degree), axis=-1)

        confusion_pairs = [confusion_class_pairs[k] for k in np.arange(0, len(top10_wrong_classes)*10, 10)]
        print("Top 10 wrong classes", top10_wrong_classes)
        print("Confusion degree entropy", wrong_degree_entropy)

        confusion_degree = confusion_degree_matrix[np.arange(len(top10_wrong_classes)), confusion_classes_ind[:, 0]]
        print("Confusion pair (top1 neighboring class only)", confusion_pairs)
        print("Confusion pair (top1 neighboring class only) degree", confusion_degree)
        return confusion_pairs


    def cache_grad_loss_train_all(self, theta, theta_hat, pair_idx):
        l_prev, l_cur = loss_change_train(self.model, self.criterion, self.dl_tr, theta, theta_hat)
        if self.measure == 'confusion':
            grad_loss = {'l_prev': l_prev, 'l_cur': l_cur}
            with open("Influential_data/{}_{}_confusion_grad4trainall_testpair{}.pkl".format(self.dataset_name, self.loss_type, pair_idx), "wb") as fp:  # Pickling
                pickle.dump(grad_loss, fp)

        elif self.measure == 'intravar':
            grad_loss = {'l_prev': l_prev, 'l_cur': l_cur}
            with open("Influential_data/{}_{}_intravar_grad4trainall_testcls{}.pkl".format(self.dataset_name, self.loss_type, pair_idx), "wb") as fp:  # Pickling
                pickle.dump(grad_loss, fp)

    # def cache_grad_loss_train_all_worse(self, theta, theta_hat, pair_idx):
    #     l_prev, l_cur = loss_change_train(self.model, self.criterion, self.dl_tr, theta, theta_hat)
    #     if self.measure == 'confusion':
    #         grad_loss = {'l_prev': l_prev, 'l_cur': l_cur}
    #         with open("Influential_data/{}_{}_confusion_grad4trainall_testpair{}_worse.pkl".format(self.dataset_name, self.loss_type, pair_idx), "wb") as fp:  # Pickling
    #             pickle.dump(grad_loss, fp) # FIXME
    #
    #     elif self.measure == 'intravar':
    #         grad_loss = {'l_prev': l_prev, 'l_cur': l_cur}
    #         with open("Influential_data/{}_{}_intravar_grad4trainall_testcls{}_worse.pkl".format(self.dataset_name, self.loss_type, pair_idx), "wb") as fp:  # Pickling
    #             pickle.dump(grad_loss, fp)

    # def theta_grad_ascent(self, classes, n=50, lr=0.001):
    #     theta_orig = self.model.module[-1].weight.data
    #     if self.measure == 'confusion':
    #         torch.cuda.empty_cache()
    #         for pair in classes:
    #             confusion_degree_orig, _ = grad_confusion(self.model, self.dl_ev, pair[0], pair[1])
    #             confusion_degree_orig = confusion_degree_orig.detach()
    #             self.model.module[-1].weight.data = theta_orig
    #             theta = theta_orig.clone()
    #             for _ in range(n):
    #                 confusion_degree, v = grad_confusion(self.model, self.dl_ev, pair[0], pair[1]) # dt2/dtheta
    #                 confusion_degree = confusion_degree.detach()
    #                 v = v[0].detach()
    #                 if confusion_degree_orig.item() - confusion_degree.item() >= 0.5: # FIXME: threshold selection
    #                     break
    #                 theta_new = theta - lr * v # gradient descent
    #                 theta = theta_new
    #                 self.model.module[-1].weight.data = theta
    #
    #             theta_dict = {'theta': theta_orig, 'theta_hat': theta}
    #             torch.save(theta_dict, "Influential_data/{}_{}_confusion_theta_test_{}_{}_worse.pth".format(self.dataset_name, self.loss_type, pair[0], pair[1])) # FIXME
    #
    #     self.model.module[-1].weight.data = theta_orig

    def theta_grad_descent(self, classes, n=50, lr=4e-5):
        theta_orig = self.model.module[-1].weight.data
        if self.measure == 'confusion':
            torch.cuda.empty_cache()
            for pair in classes:
                # D = calc_inter_dist_pair(IS.testing_embedding[IS.testing_label == pair[0]],
                #                          IS.testing_embedding[IS.testing_label == pair[1]])
                # print("Original", D.item())

                inter_dist_orig, _ = grad_interdist(self.model, self.dl_ev, pair[0], pair[1])
                inter_dist_orig = inter_dist_orig.detach()

                self.model.module[-1].weight.data = theta_orig
                theta = theta_orig.clone()
                for _ in range(n):
                    inter_dist, v = grad_interdist(self.model, self.dl_ev, pair[0], pair[1]) # dt2/dtheta
                    inter_dist = inter_dist.detach()
                    v = v[0].detach()
                    print(inter_dist.item())
                    if inter_dist.item() - inter_dist_orig.item() >= 5.: # FIXME: threshold selection
                        break
                    theta_new = theta + lr * v # gradient ascent
                    theta = theta_new
                    self.model.module[-1].weight.data = theta

                theta_dict = {'theta': theta_orig, 'theta_hat': theta}
                torch.save(theta_dict, "Influential_data/{}_{}_confusion_theta_test_{}_{}.pth".format(self.dataset_name, self.loss_type, pair[0], pair[1])) # FIXME

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
                    print(intravar.item())
                    if intravar_orig.item() - intravar.item() >= 5.: # FIXME: threshold selection
                        break
                    theta_new = theta - lr * v # gradient descent
                    theta = theta_new
                    self.model.module[-1].weight.data = theta

                theta_dict = {'theta': theta_orig, 'theta_hat': theta}
                torch.save(theta_dict, "Influential_data/{}_{}_intravar_theta_test_{}.pth".format(self.dataset_name, self.loss_type, cls))

        self.model.module[-1].weight.data = theta_orig


    def run_sample(self, pair_idx):
        if self.measure == 'confusion':
            confusion_class_pairs = self.get_confusion_class_pairs()
            pair = confusion_class_pairs[pair_idx]
            self.viz_2cls(5, self.dl_ev, self.testing_label, pair[0], pair[1])  # visualize confusion classes

            with open("Influential_data/{}_{}_confusion_grad4trainall_testpair{}.pkl".format(self.dataset_name, self.loss_type, pair_idx), "rb") as fp:  # Pickling
                grad4train = pickle.load(fp)

        elif self.measure == 'intravar':
            confusion_class_pairs = self.get_confusion_class_pairs()
            scattered_classes = [x[0] for x in confusion_class_pairs]
            scatter_cls = scattered_classes[pair_idx]
            self.viz_cls(5, self.dl_ev, self.testing_label, scatter_cls)

            with open("Influential_data/{}_{}_intravar_grad4trainall_testcls{}.pkl".format(self.dataset_name, self.loss_type, pair_idx), "rb") as fp:  # Pickling
                grad4train = pickle.load(fp)

        else:
            raise NotImplementedError

        influence_values = calc_influential_func_sample(grad4train)
        influence_values = np.asarray(influence_values)
        training_sample_by_influence = influence_values.argsort()  # ascending
        self.viz_sample(training_sample_by_influence[:10])  # helpful
        self.viz_sample(training_sample_by_influence[-10:])  # harmful

        np.save("Influential_data/{}_{}_helpful_{}_testcls{}".format(self.dataset_name, self.loss_type, self.measure, pair_idx),
                training_sample_by_influence[:500])
        np.save("Influential_data/{}_{}_harmful_{}_testcls{}".format(self.dataset_name, self.loss_type, self.measure, pair_idx),
                training_sample_by_influence[-500:])

    # def run_sample_worse(self, pair_idx):
    #     if self.measure == 'confusion':
    #         confusion_class_pairs = self.get_confusion_class_pairs()
    #         pair = confusion_class_pairs[pair_idx]
    #         self.viz_2cls(5, self.dl_ev, self.testing_label, pair[0], pair[1])  # visualize confusion classes
    #
    #         with open("Influential_data/{}_{}_confusion_grad4trainall_testpair{}_worse.pkl".format(self.dataset_name, self.loss_type, pair_idx), "rb") as fp:  # Pickling
    #             grad4train = pickle.load(fp)
    #     else:
    #         raise NotImplementedError
    #
    #     influence_values = calc_influential_func_sample(grad4train)
    #     influence_values = np.asarray(influence_values)
    #     training_sample_by_influence = influence_values.argsort()  # ascending
    #     theta1_helpful = np.load("Influential_data/{}_{}_helpful_{}_testcls{}.npy".format(self.dataset_name, self.loss_type, self.measure, pair_idx)) # help to deconfuse -> helpful
    #     theta1_harmful = np.load("Influential_data/{}_{}_harmful_{}_testcls{}.npy".format(self.dataset_name, self.loss_type, self.measure, pair_idx))
    #     theta2_harmful = training_sample_by_influence[:500]  # help to confuse -> harmful
    #     theta2_helpful = training_sample_by_influence[-500:]
    #     theta12_helpful = list(set(theta1_helpful).intersection(set(theta2_helpful)))
    #     theta12_harmful = list(set(theta1_harmful).intersection(set(theta2_harmful)))
    #     print("Helpful intersection: ", len(theta12_helpful))
    #     print("Harmful intersection: ", len(theta12_harmful))
    #     np.save("Influential_data/{}_{}_helpfulintersection_{}_testcls{}".format(self.dataset_name, self.loss_type, self.measure, pair_idx),
    #             theta12_helpful)
    #     np.save("Influential_data/{}_{}_harmfulintersection_{}_testcls{}".format(self.dataset_name, self.loss_type, self.measure, pair_idx),
    #             theta12_harmful)

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

    def viz_sample(self, indices):
        classes = self.train_label[indices]
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            img = read_image(self.dl_tr.dataset.im_paths[indices[i]])
            plt.imshow(to_pil_image(img))
            plt.title('Class = {}'.format(classes[i]))
        plt.show()

if __name__ == '__main__':

    dataset_name = 'cub'
    loss_type = 'ProxyNCA_pfix'
    # loss_type = 'ProxyNCA_pfix_confusion_111_110_reverse'
    config_name = 'cub'
    sz_embedding = 512
    seed = 4
    measure = 'intravar'
    epoch = 40
    # epoch = 1
    test_crop = True

    IS = InfluentialSample(dataset_name, seed, loss_type, config_name, measure, test_crop, sz_embedding, epoch)

    '''Other: get t statistic for two specific classes'''
    # i = 111; j = 110
    # feat_cls1 = IS.testing_embedding[IS.testing_label == i]
    # feat_cls2 = IS.testing_embedding[IS.testing_label == j]
    # confusion = calc_inter_dist(feat_cls1, feat_cls2)  # get t instead of t^2
    # print(confusion.item())
    # exit()

    # testing_embedding, testing_label, testing_indices = predict_batchwise(IS.model, IS.dl_ev)
    # feat_cls1 = testing_embedding[testing_label == i]
    # feat_cls2 = testing_embedding[testing_label == j]
    # confusion = calc_confusion(feat_cls1, feat_cls2, sqrt=True)  # get t instead of t^2
    # print(confusion.item())

    '''Other: get intra-class variance for a specific class'''
    # i = 116
    # feat_cls = IS.testing_embedding[IS.testing_label == i]
    # intra_var = calc_intravar(feat_cls)
    # print(intra_var.item())

    '''Step 1: Cache all confusion gradient to parameters'''
    confusion_class_pairs = IS.get_confusion_class_pairs()
    scatter_classes = [x[0] for x in confusion_class_pairs]

    # For theta1
    # IS.theta_grad_descent([confusion_class_pairs[k] for k in [0, 8, 9]])
    # exit()
    # For theta2
    # IS.theta_grad_ascent(confusion_class_pairs)
    # exit()

    IS.theta_grad_descent(scatter_classes)
    exit()

    '''Step 2: Cache training class loss changes'''
    # For theta1
    # pair_idx = 9 # iterate over all pairs
    # i = confusion_class_pairs[pair_idx][0]; j = confusion_class_pairs[pair_idx][1]
    # theta_dict = torch.load("Influential_data/{}_{}_confusion_theta_test_{}_{}.pth".format(IS.dataset_name, IS.loss_type, i, j))
    # theta = theta_dict['theta']
    # theta_hat = theta_dict['theta_hat']
    # IS.cache_grad_loss_train_all(theta, theta_hat, pair_idx)
    # exit()

    # For theta2
    # pair_idx = 9 # iterate over all pairs
    # i = confusion_class_pairs[pair_idx][0]; j = confusion_class_pairs[pair_idx][1]
    # theta_dict = torch.load("Influential_data/{}_{}_confusion_theta_test_{}_{}_worse.pth".format(IS.dataset_name, IS.loss_type, i, j))
    # theta = theta_dict['theta']
    # theta_hat = theta_dict['theta_hat']
    # IS.cache_grad_loss_train_all_worse(theta, theta_hat, pair_idx)
    # exit()

    # cls_idx = 7
    # i = scatter_classes[cls_idx]
    # theta_dict = torch.load("Influential_data/{}_{}_intravar_theta_test_{}.pth".format(IS.dataset_name, IS.loss_type, i))
    # theta = theta_dict['theta']
    # theta_hat = theta_dict['theta_hat']
    # IS.cache_grad_loss_train_all(theta, theta_hat, cls_idx)
    # exit()

    '''Step 3: Calc influence functions'''
    # For theta1
    IS.run_sample(0)
    exit()

    # For theta2
    # IS.run_sample_worse(9)
    # exit()

    '''Other: get losses for specific indices'''
    # dataset_name = 'cub'
    # loss_type = 'ProxyNCA_pfix'
    # config_name = 'cub'
    # sz_embedding = 512
    # seed = 4
    # measure = 'confusion'
    # epoch = 40
    # # #
    # IS_prev = InfluentialSample(dataset_name, seed, loss_type, config_name, measure, True, sz_embedding, epoch)
    # harmful = np.load('Influential_data/{}_{}_harmful_testcls{}.npy'.format('cub', 'ProxyNCA_pfix', '3'))
    # helpful = np.load('Influential_data/{}_{}_helpful_testcls{}.npy'.format('cub', 'ProxyNCA_pfix', '3'))
    # normal = list(set(range(len(IS_prev.dl_tr.dataset))) - set(harmful) - set(helpful))
    # helpful_loss_prev = calc_loss_train(IS_prev.model, IS_prev.dl_tr, IS_prev.criterion, helpful)
    # harmful_loss_prev = calc_loss_train(IS_prev.model, IS_prev.dl_tr, IS_prev.criterion, harmful)
    # normal_loss_prev = calc_loss_train(IS_prev.model, IS_prev.dl_tr, IS_prev.criterion, normal)
    # #
    # dataset_name = 'cub'
    # loss_type = 'ProxyNCA_pfix_confusion_sample500_117_129'
    # config_name = 'cub'
    # sz_embedding = 512
    # seed = 4
    # measure = 'confusion'
    # epoch = 10
    # # #
    # IS_cur = InfluentialSample(dataset_name, seed, loss_type, config_name, measure, True, sz_embedding, epoch)
    # helpful_loss_cur = calc_loss_train(IS_cur.model, IS_cur.dl_tr, IS_cur.criterion, helpful)
    # harmful_loss_cur = calc_loss_train(IS_cur.model, IS_cur.dl_tr, IS_cur.criterion, harmful)
    # normal_loss_cur = calc_loss_train(IS_cur.model, IS_cur.dl_tr, IS_cur.criterion, normal)
    #
    # harmful_diff = np.asarray(harmful_loss_cur) - np.asarray(harmful_loss_prev)
    # helpful_diff = np.asarray(helpful_loss_cur) - np.asarray(helpful_loss_prev)
    # normal_diff = np.asarray(normal_loss_cur) - np.asarray(normal_loss_prev)
    #
    # plt.hist(helpful_diff, label='helpful')
    # plt.show()
    #
    # plt.hist(harmful_diff, label='harmful')
    # plt.show()
    #
    # plt.hist(normal_diff, label='normal')
    # plt.show()
    # pass
    #
    #
    #
