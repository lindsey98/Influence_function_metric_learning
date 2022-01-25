import numpy as np
import torch
import torchvision
import loss
from networks import Feat_resnet50_max_n
from evaluation.pumap import prepare_data, get_wrong_indices
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.io.image import read_image
from Influence_function.ScalableIF_utils import *
from Influence_function.IF_utils import *
import pickle
from utils import predict_batchwise_debug
from collections import OrderedDict
import scipy.stats
from evaluation import assign_by_euclidian_at_k_indices
import utils

class BaseInfluenceFunction():
    def __init__(self, dataset_name, seed, loss_type, config_name,
                 test_crop=False, sz_embedding=512, epoch=40):

        self.folder = 'models/dvi_data_{}_{}_loss{}/'.format(dataset_name, seed, loss_type)
        self.model_dir = '{}/ResNet_{}_Model'.format(self.folder, sz_embedding)
        self.config_name = config_name
        self.epoch = epoch

        # load data
        self.dl_tr, self.dl_ev = prepare_data(data_name=dataset_name,
                                              config_name=self.config_name,
                                              batch_size=1,
                                              test_crop=test_crop)
        self.dataset_name = dataset_name
        self.seed = seed
        self.loss_type = loss_type
        self.sz_embedding = sz_embedding
        self.model = self._load_model()
        self.model.eval()
        self.criterion = self._load_criterion()
        self.train_embedding, self.train_label, self.testing_embedding, self.testing_label, \
            self.testing_nn_label, self.testing_nn_indices = self._load_data()
        pass

    def _load_model(self, multi_gpu=True):
        feat = Feat_resnet50_max_n()
        emb = torch.nn.Linear(2048, self.sz_embedding) # projection layer
        model = torch.nn.Sequential(feat, emb)
        weights = torch.load(
            '{}/Epoch_{}/{}_{}_trainval_{}_{}.pth'.format(self.model_dir, self.epoch, self.dataset_name, self.dataset_name, self.sz_embedding, self.seed))
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

    def _load_model_random(self, multi_gpu=True):
        feat = Feat_resnet50_max_n()
        emb = torch.nn.Linear(2048, self.sz_embedding) # projection layer
        model = torch.nn.Sequential(feat, emb)
        if multi_gpu:
            model = nn.DataParallel(model)
        model.cuda()
        return model

    def _load_criterion(self):
        proxies = torch.load('{}/Epoch_{}/proxy.pth'.format(self.model_dir, self.epoch), map_location='cpu')['proxies'].detach()
        if 'ProxyNCA_prob_orig' in self.loss_type:
            criterion = loss.ProxyNCA_prob_orig(nb_classes=self.dl_tr.dataset.nb_classes(), sz_embed=self.sz_embedding, scale=3)
        elif 'ProxyAnchor' in self.loss_type:
            criterion = loss.Proxy_Anchor(nb_classes=self.dl_tr.dataset.nb_classes(), sz_embed=self.sz_embedding)
        elif 'SoftTriple' in self.loss_type:
            criterion = loss.SoftTriple(nb_classes=self.dl_tr.dataset.nb_classes(), sz_embed=self.sz_embedding,
                                        la=20, gamma=0.1, tau=0.2, margin=0.01, K=1)
        else:
            raise NotImplementedError
        criterion.proxies.data = proxies
        criterion.cuda()
        return criterion

    def cache_embedding(self):
        embedding, label, _ = predict_batchwise_debug(self.model, self.dl_tr)
        torch.save(embedding, '{}/Epoch_{}/training_embeddings.pth'.format(self.model_dir, self.epoch))
        torch.save(label, '{}/Epoch_{}/training_labels.pth'.format(self.model_dir, self.epoch))

        testing_embedding, testing_label, _ = predict_batchwise_debug(self.model, self.dl_ev)
        torch.save(testing_embedding, '{}/Epoch_{}/testing_embeddings.pth'.format(self.model_dir, self.epoch))
        torch.save(testing_label, '{}/Epoch_{}/testing_labels.pth'.format(self.model_dir, self.epoch))

        # Caution: I only care about these pairs, re-evaluation (if any) should also be based on these pairs
        nn_indices, pred = assign_by_euclidian_at_k_indices(testing_embedding, testing_label, 1)  # predict
        torch.save(torch.from_numpy(pred.flatten()), '{}/Epoch_{}/testing_nn_labels.pth'.format(self.model_dir, self.epoch))
        np.save('{}/Epoch_{}/testing_nn_indices'.format(self.model_dir, self.epoch), nn_indices)

    def _load_data(self):
        try:
            train_embedding = torch.load('{}/Epoch_{}/training_embeddings.pth'.format(self.model_dir, self.epoch))  # high dimensional embedding
            train_label = torch.load('{}/Epoch_{}/training_labels.pth'.format(self.model_dir, self.epoch))
            testing_embedding = torch.load('{}/Epoch_{}/testing_embeddings.pth'.format(self.model_dir, self.epoch))  # high dimensional embedding
            testing_label = torch.load('{}/Epoch_{}/testing_labels.pth'.format(self.model_dir, self.epoch))
            testing_nn_label = torch.load('{}/Epoch_{}/testing_nn_labels.pth'.format(self.model_dir, self.epoch))
            testing_nn_indices = np.load('{}/Epoch_{}/testing_nn_indices.npy'.format(self.model_dir, self.epoch))

        except FileNotFoundError:
            self.cache_embedding()
            train_embedding = torch.load('{}/Epoch_{}/training_embeddings.pth'.format(self.model_dir, self.epoch))  # high dimensional embedding
            train_label = torch.load('{}/Epoch_{}/training_labels.pth'.format(self.model_dir, self.epoch))
            testing_embedding = torch.load('{}/Epoch_{}/testing_embeddings.pth'.format(self.model_dir, self.epoch))  # high dimensional embedding
            testing_label = torch.load('{}/Epoch_{}/testing_labels.pth'.format(self.model_dir, self.epoch))
            testing_nn_label = torch.load('{}/Epoch_{}/testing_nn_labels.pth'.format(self.model_dir, self.epoch))
            testing_nn_indices = np.load('{}/Epoch_{}/testing_nn_indices.npy'.format(self.model_dir, self.epoch))

        return train_embedding, train_label, testing_embedding, testing_label, testing_nn_label, testing_nn_indices

    @torch.no_grad()
    def get_train_features(self):
        self.model.eval()
        # Forward propogate up to projection layer, cache the features (testing loader)
        all_features = torch.tensor([])  # (N, 2048)
        for ct, (x, t, _) in tqdm(enumerate(self.dl_tr)):
            x = x.cuda()
            m = self.model.module[:-1](x)
            all_features = torch.cat((all_features, m.detach().cpu()), dim=0)
        return all_features

    @torch.no_grad()
    def get_features(self):
        self.model.eval()
        # Forward propogate up to projection layer, cache the features (testing loader)
        all_features = torch.tensor([])  # (N, 2048)
        for ct, (x, t, _) in tqdm(enumerate(self.dl_ev)):
            x = x.cuda()
            m = self.model.module[:-1](x)
            all_features = torch.cat((all_features, m.detach().cpu()), dim=0)
        return all_features

    def get_wrong_freq_matrix(self, top10_wrong_classes, wrong_labels, wrong_preds):
        wrong_labels = wrong_labels.detach().cpu().numpy()
        wrong_preds = wrong_preds.squeeze().detach().cpu().numpy()

        wrong_freq_matrix = np.zeros((len(top10_wrong_classes), len(self.dl_ev.dataset.classes)))
        for indi, i in enumerate(top10_wrong_classes):
            ct_wrong = np.sum(wrong_labels == i)
            for indj, j in enumerate(self.dl_ev.dataset.classes):
                if i == j:
                    continue
                ct_wrong_as_clsj = np.sum((wrong_preds == j) & (wrong_labels == i)) # wrong and predicted as j
                wrong_freq_matrix[indi][indj] = - ct_wrong_as_clsj / ct_wrong # negative for later sorting purpose

        return wrong_freq_matrix

    def get_confusion_class_pairs(self):
        _, top10_wrong_classes, wrong_labels, wrong_preds = get_wrong_indices(self.testing_embedding, self.testing_label, topk=10)

        wrong_freq_matrix = self.get_wrong_freq_matrix(top10_wrong_classes, wrong_labels, wrong_preds)
        confusion_classes_ind = np.argsort(wrong_freq_matrix, axis=-1)[:, :10]  # get top 10 classes that are frequently confused with top 10 wrong testing classes

        # Find the first index which explains >half of the wrong cases (cumulatively)
        confusion_class_degree = -1 * wrong_freq_matrix[np.repeat(np.arange(len(confusion_classes_ind)), 10),
                                                        confusion_classes_ind.flatten()].reshape(len(confusion_classes_ind), -1)
        result = np.cumsum(confusion_class_degree, -1)
        row, col = np.where(result > 0.5)
        first_index = [np.where(row == i)[0][0] if len(np.where(row == i)[0]) > 0 else 9 \
                       for i in range(len(confusion_classes_ind))]
        first_index = col[first_index]

        # Filter out those confusion class pairs
        confusion_class_pairs = [[(top10_wrong_classes[i], np.asarray(self.dl_ev.dataset.classes)[confusion_classes_ind[i][j]]) \
                                  for j in range(first_index[i] + 1)] \
                                  for i in range(confusion_classes_ind.shape[0])]

        print("Top 10 wrong classes", top10_wrong_classes)
        print("Confusion pairs", confusion_class_pairs)

        return confusion_class_pairs

    def viz_cls(self, top_bottomk, dl, label, cls):
        ind_cls = np.where(label.detach().cpu().numpy() == cls)[0]
        for i in range(top_bottomk):
            plt.subplot(1, top_bottomk, i + 1)
            img = read_image(dl.dataset.im_paths[ind_cls[i]])
            plt.imshow(to_pil_image(img))
            plt.title('Class = {}'.format(cls))
        plt.show()

    def viz_2cls(self, top_bottomk, dl, label, cls1, cls2):
        ind_cls1 = np.where(label.detach().cpu().numpy() == cls1)[0]
        ind_cls2 = np.where(label.detach().cpu().numpy() == cls2)[0]

        top_bottomk = min(top_bottomk, len(ind_cls1), len(ind_cls2))
        for i in range(top_bottomk):
            plt.subplot(2, top_bottomk, i + 1)
            img = read_image(dl.dataset.im_paths[ind_cls1[i]])
            plt.imshow(to_pil_image(img))
            plt.title('Class = {}'.format(cls1))
        for i in range(top_bottomk):
            plt.subplot(2, top_bottomk, i + 1 + top_bottomk)
            img = read_image(dl.dataset.im_paths[ind_cls2[i]])
            plt.imshow(to_pil_image(img))
            plt.title('Class = {}'.format(cls2))
        plt.show()

    def viz_sample(self, dl, indices):
        classes = [dl.dataset.ys[x] for x in indices]
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            img = read_image(dl.dataset.im_paths[indices[i]])
            plt.imshow(to_pil_image(img))
            plt.title('Class = {}'.format(classes[i]))
        plt.show()

    def viz_2sample(self, dl, ind1, ind2):
        class1 = dl.dataset.ys[ind1]
        class2 = dl.dataset.ys[ind2]

        plt.subplot(1, 2, 1)
        img = read_image(dl.dataset.im_paths[ind1])
        plt.imshow(to_pil_image(img))
        plt.title('Class = {}'.format(class1))

        plt.subplot(1, 2, 2)
        img = read_image(dl.dataset.im_paths[ind2])
        plt.imshow(to_pil_image(img))
        plt.title('Class = {}'.format(class2))
        plt.show()
        plt.close()

class MCScalableIF(BaseInfluenceFunction):
    def __int__(self, dataset_name, seed, loss_type, config_name,
                 test_crop=False, sz_embedding=512, epoch=40):
        super(MCScalableIF, self).__init__(dataset_name, seed, loss_type, config_name,
                                         test_crop=False, sz_embedding=512, epoch=40)

    def get_grad_loss_train_all(self, theta, theta_hat):
        l_prev, l_cur = loss_change_train(self.model, self.criterion, self.dl_tr, theta, theta_hat)
        grad_loss = {'l_prev': l_prev, 'l_cur': l_cur}
        return grad_loss

    def get_theta_by_newton_step(self, all_features, wrong_cls, confused_classes,
                                 theta_orig,
                                 inter_dist_orig, descent=False):

        inter_dist, v = grad_confusion(self.model, all_features, wrong_cls, confused_classes,
                            self.testing_nn_label, self.testing_label, self.testing_nn_indices) # dD/dtheta
        v = F.normalize(v[0].to(theta_orig.device).flatten(), p=2) # L2 normalization

        if not descent:
            theta_steepest = theta_orig + v # gradient ascent, deconfusion
        else:
            theta_steepest = theta_orig - v  # gradient descent, confusion

        grad_loss = self.get_grad_loss_train_all(theta_orig.reshape(self.model.module[-1].weight.data.shape[0],
                                                                    self.model.module[-1].weight.data.shape[1]),
                                                 theta_steepest.reshape(self.model.module[-1].weight.data.shape[0],
                                                                    self.model.module[-1].weight.data.shape[1]))
        deltaD = inter_dist - inter_dist_orig # scalar
        l_prev = grad_loss['l_prev']; l_cur = grad_loss['l_cur']
        deltaL = np.stack(l_cur) - np.stack(l_prev) # (N, )
        return deltaD, deltaL, theta_steepest

    def get_theta_by_orthogonalization(self, num_thetas, prev_thetas,
                                       all_features, wrong_cls, confused_classes,
                                     theta_orig,
                                     inter_dist_orig):
        new_thetas = torch.randn((num_thetas, len(prev_thetas[0])), requires_grad=True)
        prev_thetas.to(new_thetas.device)
        prev_thetas.requires_grad = False
        deltaD_deltaL = []

        _optimizer = torch.optim.Adam(params={new_thetas}, lr=0.1)
        for _ in tqdm(range(50), desc="Orthogonalizing the thetas"):
            meaninter, varinter, means2prev, var2prev = utils.inter_dist(new_thetas, prev_thetas)
            _loss = meaninter + varinter + means2prev + var2prev
            _optimizer.zero_grad()
            _loss.backward()
            _optimizer.step()

        new_thetas = F.normalize(new_thetas, p=2, dim=-1).detach().cpu()

        for this_theta in new_thetas:
            model_copy = self._load_model()
            model_copy.module[-1].weight.data = this_theta
            inter_dist, _ = grad_confusion(model_copy, all_features, wrong_cls, confused_classes, self.testing_nn_label, self.testing_label, self.testing_nn_indices)
            model_copy.module[-1].weight.data = theta_orig
            grad_loss = self.get_grad_loss_train_all(theta_orig.reshape(model_copy.module[-1].weight.data.shape[0],
                                                                        model_copy.module[-1].weight.data.shape[1]),
                                                     this_theta.reshape(model_copy.module[-1].weight.data.shape[0],
                                                                        model_copy.module[-1].weight.data.shape[1]))
            deltaD = inter_dist - inter_dist_orig  # scalar
            l_prev = grad_loss['l_prev']; l_cur = grad_loss['l_cur']
            deltaL = np.stack(l_cur) - np.stack(l_prev)  # (N, )
            deltaD_deltaL.append(deltaD / (deltaL + 1e-8))

        return new_thetas, deltaD_deltaL

    def MC_estimate(self, pair, num_thetas=2):

        theta_orig = self.model.module[-1].weight.data.flatten()
        torch.cuda.empty_cache()
        all_features = self.get_features() # (N, 2048)
        deltaD_deltaL = []
        theta_list = torch.tensor([])

        wrong_cls = pair[0][0]
        confused_classes = [x[1] for x in pair]
        inter_dist_orig, _ = grad_confusion(self.model, all_features, wrong_cls, confused_classes,
                                self.testing_nn_label, self.testing_label, self.testing_nn_indices) # dD/dtheta

        for kk in range(num_thetas):
            if kk == 0:
                ''' first theta is the steepest ascent direction '''
                deltaD, deltaL, theta_new = self.get_theta_by_newton_step(all_features, wrong_cls, confused_classes,
                                                               theta_orig,
                                                               inter_dist_orig, descent=False)
                deltaD_deltaL.append(deltaD / (deltaL + 1e-8))
                theta_list = torch.cat([theta_list, theta_new])

            elif kk == 1:
                '''second theta is the steepest descent direction'''
                deltaD, deltaL, theta_new = self.get_theta_by_newton_step(all_features, wrong_cls, confused_classes,
                                                               theta_orig,
                                                               inter_dist_orig, descent=True)
                deltaD_deltaL.append(deltaD / (deltaL + 1e-8))
                theta_list = torch.cat([theta_list, theta_new])

            else:
                '''If more thetas are needed'''
                theta_new, deltaD_deltaL_more = self.get_theta_by_orthogonalization(num_thetas=num_thetas-2, prev_thetas=theta_list)
                theta_list = torch.cat([theta_list, theta_new])
                deltaD_deltaL.extend(deltaD_deltaL_more)
                break

        self.model.module[-1].weight.data = theta_orig
        return theta_list, deltaD_deltaL


class ScalableIF(BaseInfluenceFunction):
    def __int__(self, dataset_name, seed, loss_type, config_name,
                 test_crop=False, sz_embedding=512, epoch=40):
        super(ScalableIF, self).__init__(dataset_name, seed, loss_type, config_name,
                                         test_crop=False, sz_embedding=512, epoch=40)

    def cache_grad_loss_train_all(self, theta, theta_hat, pair_idx):
        l_prev, l_cur = loss_change_train(self.model, self.criterion, self.dl_tr, theta, theta_hat)
        grad_loss = {'l_prev': l_prev, 'l_cur': l_cur}
        with open("Influential_data/{}_{}_confusion_grad4trainall_testpair{}.pkl".format(self.dataset_name, self.loss_type, pair_idx), "wb") as fp:  # Pickling
            pickle.dump(grad_loss, fp)

    def agg_get_theta(self, classes, steps=50, lr=0.001, save=True):

        theta_orig = self.model.module[-1].weight.data
        torch.cuda.empty_cache()

        all_features = self.get_features() # (N, 2048)

        for pair in classes:
            wrong_cls = pair[0][0]
            confused_classes = [x[1] for x in pair]

            # revise back the weights
            self.model.module[-1].weight.data = theta_orig
            theta = theta_orig.clone()

            # Record original inter-class distance
            inter_dist_orig, _ = grad_confusion(self.model, all_features, wrong_cls, confused_classes,
                                                self.testing_nn_label, self.testing_label, self.testing_nn_indices) # dD/dtheta
            print("Original confusion: ", inter_dist_orig)

            # Optimization
            for _ in range(steps):
                inter_dist, v = grad_confusion(self.model, all_features, wrong_cls, confused_classes,
                                               self.testing_nn_label, self.testing_label, self.testing_nn_indices) # dD/dtheta
                print("Confusion: ", inter_dist)
                if inter_dist - inter_dist_orig >= 1.: # FIXME: stopping criteria threshold selection
                    break
                theta_new = theta + lr * v[0].to(theta.device) # gradient ascent
                theta = theta_new
                self.model.module[-1].weight.data = theta

            theta_dict = {'theta': theta_orig, 'theta_hat': theta}
            if save:
                torch.save(theta_dict, "Influential_data/{}_{}_confusion_theta_test_{}.pth".format(self.dataset_name, self.loss_type, wrong_cls))

        self.model.module[-1].weight.data = theta_orig
        if not save:
            return theta_dict
        else:
            return

    def agg_influence_func(self, pair_idx, save=True):
        confusion_class_pairs = self.get_confusion_class_pairs()
        pair = confusion_class_pairs[pair_idx]
        self.viz_2cls(5, self.dl_ev, self.testing_label, pair[0][0], pair[0][1])  # visualize confusion classes

        with open("Influential_data/{}_{}_confusion_grad4trainall_testpair{}.pkl".format(self.dataset_name, self.loss_type, pair_idx), "rb") as fp:  # Pickling
            grad4train = pickle.load(fp)

        influence_values = calc_influential_func_sample(grad4train)
        influence_values = np.asarray(influence_values)
        training_sample_by_influence = influence_values.argsort()  # ascending
        self.viz_sample(self.dl_tr, training_sample_by_influence[:10])  # helpful
        self.viz_sample(self.dl_tr, training_sample_by_influence[-10:])  # harmful

        helpful_indices = np.where(influence_values < 0)[0] # cache all helpful
        harmful_indices = np.where(influence_values > 0)[0] # cache all harmful
        if save:
            np.save("Influential_data/{}_{}_helpful_testcls{}".format(self.dataset_name, self.loss_type, pair_idx),
                    helpful_indices)
            np.save("Influential_data/{}_{}_harmful_testcls{}".format(self.dataset_name, self.loss_type, pair_idx),
                    harmful_indices)
        else:
            return helpful_indices, harmful_indices, influence_values

    def single_get_theta(self, theta_orig, all_features, wrong_indices, confuse_indices):
        # Revise back the weights
        self.model.module[-1].weight.data = theta_orig
        theta = theta_orig.clone()

        # Record original inter-class distance
        inter_dist_orig, _ = grad_confusion_pair(self.model, all_features, wrong_indices, confuse_indices)  # dD/dtheta
        print("Original confusion: ", inter_dist_orig)

        # Optimization
        for _ in range(50):
            inter_dist, v = grad_confusion_pair(self.model, all_features, wrong_indices, confuse_indices)  # dD/dtheta
            print("Confusion: ", inter_dist)
            if inter_dist - inter_dist_orig >= 1.:  # FIXME: stopping criteria threshold selection
                break
            theta_new = theta + 0.001 * v[0].to(theta.device)  # gradient ascent
            theta = theta_new
            self.model.module[-1].weight.data = theta

        self.model.module[-1].weight.data = theta_orig
        return theta

    def single_influence_func(self, all_features, wrong_indices, confuse_indices):

        '''Step 1: All confusion gradient to parameters'''
        theta_orig = self.model.module[-1].weight.data
        torch.cuda.empty_cache()
        theta = self.single_get_theta(theta_orig, all_features, wrong_indices, confuse_indices)

        '''Step 2: Training class loss changes'''
        l_prev, l_cur = loss_change_train(self.model, self.criterion, self.dl_tr, theta_orig, theta)
        grad_loss = {'l_prev': l_prev, 'l_cur': l_cur}

        '''Step 3: Calc influence functions'''
        # sanity check self.viz_2sample(self.dl_ev, wrong_indices[0], confuse_indices[0])
        influence_values = calc_influential_func_sample(grad_loss)
        influence_values = np.asarray(influence_values)
        training_sample_by_influence = influence_values.argsort()  # ascending
        print('Proportion of negative change: ', np.sum(influence_values < 0) / len(influence_values))
        print('Proportion of zero change: ', np.sum(influence_values == 0) / len(influence_values))
        print('Proportion of positive change: ', np.sum(influence_values > 0) / len(influence_values))
        # sanity check self.viz_sample(self.dl_tr, training_sample_by_influence[:10])  # helpful
        # sanity check self.viz_sample(self.dl_tr, training_sample_by_influence[-10:])  # harmful
        return training_sample_by_influence, influence_values

class OrigIF(BaseInfluenceFunction):
    def __int__(self, dataset_name, seed, loss_type, config_name,
                test_crop=False, sz_embedding=512, epoch=40):

        super(BaseInfluenceFunction, self).__init__(dataset_name, seed, loss_type, config_name,
                                         test_crop=False, sz_embedding=512, epoch=40)

    def single_influence_func_orig(self, train_features, test_features, wrong_indices, confuse_indices):
        inter_dist_pair, v = grad_confusion_pair(self.model, test_features, wrong_indices, confuse_indices)
        ihvp = inverse_hessian_product(self.model, self.criterion, v, self.dl_tr, scale=500, damping=0.01)
        influence_values = calc_influential_func_orig(IS=self, train_features=train_features, inverse_hvp_prod=ihvp)
        influence_values = np.asarray(influence_values).flatten()

        return influence_values
