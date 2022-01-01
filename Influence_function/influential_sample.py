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
from Influence_function.influence_function import *
import pickle
from utils import predict_batchwise
from collections import OrderedDict
import scipy.stats
from evaluation import assign_by_euclidian_at_k_indices
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

class InfluentialSample():
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
        self.criterion = self._load_criterion()
        self.train_embedding, self.train_label, self.testing_embedding, self.testing_label, \
            self.testing_nn_label, self.testing_nn_indices = self._load_data()

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
    def get_features(self):
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

    def cache_grad_loss_train_all(self, theta, theta_hat, pair_idx):
        l_prev, l_cur = loss_change_train(self.model, self.criterion, self.dl_tr, theta, theta_hat)
        grad_loss = {'l_prev': l_prev, 'l_cur': l_cur}
        with open("Influential_data/{}_{}_confusion_grad4trainall_testpair{}_threshold50.pkl".format(self.dataset_name, self.loss_type, pair_idx), "wb") as fp:  # Pickling
            pickle.dump(grad_loss, fp)

    def theta_grad_ascent(self, classes, steps=50, lr=4e-5):

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
                if inter_dist - inter_dist_orig >= 50.: # FIXME: stopping criteria threshold selection
                    break
                theta_new = theta + lr * v[0].to(theta.device) # gradient ascent
                theta = theta_new
                self.model.module[-1].weight.data = theta

            theta_dict = {'theta': theta_orig, 'theta_hat': theta}
            torch.save(theta_dict, "Influential_data/{}_{}_confusion_theta_test_{}_threshold50.pth".format(self.dataset_name, self.loss_type, wrong_cls))

        self.model.module[-1].weight.data = theta_orig


    def run_sample(self, pair_idx):
        confusion_class_pairs = self.get_confusion_class_pairs()
        pair = confusion_class_pairs[pair_idx]
        self.viz_2cls(5, self.dl_ev, self.testing_label, pair[0][0], pair[0][1])  # visualize confusion classes

        with open("Influential_data/{}_{}_confusion_grad4trainall_testpair{}_threshold50.pkl".format(self.dataset_name, self.loss_type, pair_idx), "rb") as fp:  # Pickling
            grad4train = pickle.load(fp)

        influence_values = calc_influential_func_sample(grad4train)
        influence_values = np.asarray(influence_values)
        training_sample_by_influence = influence_values.argsort()  # ascending
        self.viz_sample(training_sample_by_influence[:10])  # helpful
        self.viz_sample(training_sample_by_influence[-10:])  # harmful

        np.save("Influential_data/{}_{}_helpful_testcls{}_threshold50".format(self.dataset_name, self.loss_type, pair_idx),
                training_sample_by_influence[:500])
        np.save("Influential_data/{}_{}_harmful_testcls{}_threshold50".format(self.dataset_name, self.loss_type, pair_idx),
                training_sample_by_influence[-500:])

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

    # dataset_name = 'cars'
    # loss_type = 'ProxyNCA_pfix'
    # config_name = 'cars'
    # sz_embedding = 512
    # seed = 4
    # epoch = 40
    # test_crop = False

    # dataset_name = 'sop'
    # loss_type = 'ProxyNCA_pfix_var'
    # config_name = 'sop'
    # sz_embedding = 512
    # seed = 2
    # epoch = 40
    # test_crop = True

    dataset_name = 'inshop'
    loss_type = 'ProxyNCA_pfix_var_complicate'
    config_name = 'inshop'
    sz_embedding = 512
    seed = 3
    epoch = 40
    test_crop = True

    IS = InfluentialSample(dataset_name, seed, loss_type, config_name, test_crop, sz_embedding, epoch)

    '''Other: get confusion (before VS after)'''
    # FIXME: inter class distance should be computed based on original confusion pairs
    # FIXME: confusion class pairs is computed with original weights, then we do weight reload
    features = IS.get_features()
    confusion_class_pairs = IS.get_confusion_class_pairs()
    for pair_idx in range(len(confusion_class_pairs)):
        print('Pair index', pair_idx)
        wrong_cls = confusion_class_pairs[pair_idx][0][0]
        confuse_classes = [x[1] for x in confusion_class_pairs[pair_idx]]
        IS.model = IS._load_model() # reload the original weights

        inter_dist_orig, _ = grad_confusion(IS.model, features, wrong_cls, confuse_classes,
                                            IS.testing_nn_label, IS.testing_label, IS.testing_nn_indices)
        print("Original inter-class distance: ", inter_dist_orig)

        # reload weights as new
        IS.model.load_state_dict(torch.load(
                'models/dvi_data_{}_{}_loss{}_{}_{}/ResNet_512_Model/Epoch_{}/{}_{}_trainval_{}_{}.pth'.format(dataset_name, seed,
                                 'ProxyNCA_pfix_confusion_{}_threshold50'.format(wrong_cls),
                                 2, 0,
                                 1, dataset_name,
                                 dataset_name, 512, seed)))
        # IS.model = IS._load_model_random()
        inter_dist_after, _ = grad_confusion(IS.model, features, wrong_cls, confuse_classes,
                                             IS.testing_nn_label, IS.testing_label, IS.testing_nn_indices)
        print("After inter-class distance: ", inter_dist_after)
    exit()

    '''Step 1: Cache all confusion gradient to parameters'''
    confusion_class_pairs = IS.get_confusion_class_pairs()
    IS.theta_grad_ascent(confusion_class_pairs)

    '''Step 2: Cache training class loss changes'''
    for cls_idx in range(len(confusion_class_pairs)):
        i = confusion_class_pairs[cls_idx][0][0]
        theta_dict = torch.load("Influential_data/{}_{}_confusion_theta_test_{}_threshold50.pth".format(IS.dataset_name, IS.loss_type, i))
        theta = theta_dict['theta']
        theta_hat = theta_dict['theta_hat']
        IS.cache_grad_loss_train_all(theta, theta_hat, cls_idx)

    '''Step 3: Calc influence functions'''
    for cls_idx in range(len(confusion_class_pairs)):
        IS.run_sample(cls_idx)
    exit()

