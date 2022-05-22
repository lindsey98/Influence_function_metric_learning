
import os
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torchvision.io.image import read_image
from Influence_function.influence_function import MCScalableIF
from explanation.CAM_methods import *
from Influence_function.EIF_utils import *
from Influence_function.IF_utils import *
from utils import overlay_mask
from evaluation import assign_by_euclidian_at_k_indices
import sklearn
import pickle
from utils import evaluate
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR, https://github.com/PatrickHua/SimSiam/blob/main/tools/knn_monitor.py
def kNN_label_pred(query_indices, embeddings, labels, nb_classes, knn_k):

    distances = sklearn.metrics.pairwise.pairwise_distances(embeddings) # (N, N)
    indices = np.argsort(distances, axis=1)[:, 1: knn_k + 1] # (N, knn_k)
    query_nn_indices = indices[query_indices] # (B, knn_k)
    query_nn_labels = torch.gather(labels.expand(query_indices.shape[0], -1),
                                   dim=-1,
                                   index=torch.from_numpy(query_nn_indices)) # (B, knn_k)

    query2nn_dist = distances[np.repeat(query_indices, knn_k), query_nn_indices.flatten()] # (B*knn_k, )
    query2nn_dist = query2nn_dist.reshape(len(query_indices), -1) # (B, knn_k)
    query2nn_dist_exp = (-torch.from_numpy(query2nn_dist)).exp() # (B, knn_k)

    one_hot_label = torch.zeros(query_indices.shape[0] * knn_k, nb_classes, device=query_nn_labels.device) # (B*knn_k, C)
    one_hot_label = one_hot_label.scatter(dim=-1, index=query_nn_labels.view(-1, 1).type(torch.int64), value=1.0) # (B*knn_k, C)

    raw_pred_scores = torch.sum(one_hot_label.view(query_indices.shape[0], -1, nb_classes) * query2nn_dist_exp.unsqueeze(dim=-1), dim=1) # (B, C)
    pred_scores = raw_pred_scores / torch.sum(raw_pred_scores, dim=-1, keepdim=True) # (B, C)
    pred_labels = torch.argsort(pred_scores, dim=-1, descending=True) # (B, C)
    return pred_labels, pred_scores

class SampleRelabel(MCScalableIF):
    def __init__(self, dataset_name, seed, loss_type, config_name,
                 test_crop=False, sz_embedding=512, epoch=40):

        super().__init__(dataset_name, seed, loss_type, config_name,
                          test_crop, sz_embedding, epoch)

    def getNN_indices(self, embedding, label):
        # global 1st NN
        nn_indices, nn_label = assign_by_euclidian_at_k_indices(embedding, label, 1)
        nn_indices, nn_label = nn_indices.flatten(), nn_label.flatten()

        # Same class 1st NN
        chunk_size = 1000 # you need to chunk because the number of samples is too large
        num_chunks = math.ceil(len(embedding) / chunk_size)
        distances = torch.tensor([])
        for i in tqdm(range(0, num_chunks)):
            chunk_indices = [chunk_size * i, min(len(embedding), chunk_size * (i + 1))]
            chunk_X = embedding[chunk_indices[0]:chunk_indices[1], :]
            distance_mat = torch.from_numpy(sklearn.metrics.pairwise.pairwise_distances(embedding, chunk_X))
            distances = torch.cat((distances, distance_mat), dim=-1)
        distances = distances.detach().cpu().numpy()

        diff_cls_mask = (label[:, None] != label).detach().cpu().numpy().nonzero()
        distances[diff_cls_mask[0], diff_cls_mask[1]] = distances.max() + 1
        nn_indices_same_cls = np.argsort(distances, axis=1)[:, 1] # get NN among all same-class samples

        return nn_indices, nn_label, nn_indices_same_cls

    def calc_relabel_dict(self, lookat_harmful,
                          harmful_indices, helpful_indices,
                          base_dir, pair_ind1, pair_ind2):

        assert isinstance(lookat_harmful, bool)
        if lookat_harmful:
            top_indices = harmful_indices  # top_harmful_indices = influence_values.argsort()[-50:]
        else:
            top_indices = helpful_indices

        relabel_dict = {}
        unique_labels, unique_counts = torch.unique(self.train_label, return_counts=True)
        median_shots_percls = unique_counts.median().item()
        _, prob_relabel = kNN_label_pred(query_indices=top_indices, embeddings=self.train_embedding, labels=self.train_label,
                                         nb_classes=self.dl_tr.dataset.nb_classes(), knn_k=median_shots_percls)

        for kk in range(len(top_indices)):
            relabel_dict[top_indices[kk]] = prob_relabel[kk].detach().cpu().numpy()

        with open('./{}/Allrelabeldict_{}_{}_soft_knn.pkl'.format(base_dir, pair_ind1, pair_ind2), 'wb') as handle:
            pickle.dump(relabel_dict, handle)



if __name__ == '__main__':

    loss_type = 'ProxyNCA_prob_orig'; sz_embedding = 512; epoch = 40; test_crop = False
    dataset_name = 'cub';  config_name = 'cub'; seed = 0
    # dataset_name = 'cars'; config_name = 'cars'; seed = 3
    # dataset_name = 'inshop'; config_name = 'inshop'; seed = 4

    IS = SampleRelabel(dataset_name, seed, loss_type, config_name, test_crop)

    '''Analyze pairs with generalization error'''
    '''Step 1: Visualize all pairs (confuse), Find interesting pairs'''
    test_nn_indices, test_nn_label, test_nn_indices_same_cls = IS.getNN_indices(IS.testing_embedding, IS.testing_label)
    wrong_indices = (test_nn_label != IS.testing_label.detach().cpu().numpy().flatten()).nonzero()[0]
    confuse_indices = test_nn_indices[wrong_indices]
    print(len(confuse_indices))
    assert len(wrong_indices) == len(confuse_indices)

    # Same class 1st NN
    wrong_test_nn_indices_same_cls = test_nn_indices_same_cls[wrong_indices]
    assert len(wrong_indices) == len(wrong_test_nn_indices_same_cls)

    IS.vis_pairs(wrong_indices, confuse_indices, wrong_test_nn_indices_same_cls,
                 IS.dl_ev, base_dir='Confuse_Vis')
    exit()

    '''Step 2: Identify influential training points for a specific pair'''
    lines = open('explanation/{}_{}'.format(IS.dataset_name, 'ModelD_HumanS_pairs')).readlines()
    lookat_harmful = False
    relabel_method = 'soft_knn'
    base_dir = 'Confuse_pair_influential_data/{}'.format(IS.dataset_name)
    os.makedirs(base_dir, exist_ok=True)
    all_features = IS.get_test_features()
    for line in tqdm(lines):
        pair_ind1, pair_ind2 = line.strip().split(',')
        pair_ind1, pair_ind2 = int(pair_ind1), int(pair_ind2)
        if not os.path.exists('./{}/All_influence_{}_{}.npy'.format(base_dir, pair_ind1, pair_ind2)):
            # sanity check: # IS.viz_2sample(IS.dl_ev, pair_ind1, pair_ind2)
            training_sample_by_influence, influence_values = IS.MC_estimate_forpair(all_features, [pair_ind1], [pair_ind2])
            helpful_indices = np.where(influence_values < 0)[0]
            harmful_indices = np.where(influence_values > 0)[0]
            np.save('./{}/Allhelpful_indices_{}_{}'.format(base_dir, pair_ind1, pair_ind2), helpful_indices)
            np.save('./{}/Allharmful_indices_{}_{}'.format(base_dir, pair_ind1, pair_ind2), harmful_indices)
            np.save('./{}/All_influence_{}_{}'.format(base_dir, pair_ind1, pair_ind2), influence_values)
        else:
            helpful_indices = np.load('./{}/Allhelpful_indices_{}_{}.npy'.format(base_dir, pair_ind1, pair_ind2))
            harmful_indices = np.load('./{}/Allharmful_indices_{}_{}.npy'.format(base_dir, pair_ind1, pair_ind2))
            influence_values = np.load('./{}/All_influence_{}_{}.npy'.format(base_dir, pair_ind1, pair_ind2))

        # Global 1st NN
        train_nn_indices, train_nn_label, train_nn_indices_same_cls = IS.getNN_indices(IS.train_embedding, IS.train_label)
        assert len(train_nn_indices_same_cls) == len(train_nn_indices)
        assert len(IS.train_label) == len(train_nn_indices)

        '''Step 3: Save harmful indices as well as its neighboring indices'''
        IS.calc_relabel_dict(lookat_harmful=lookat_harmful,
                             harmful_indices=harmful_indices, helpful_indices=helpful_indices,
                             base_dir=base_dir, pair_ind1=pair_ind1, pair_ind2=pair_ind2)
    exit()

    '''Visualize harmful/helpful training '''
    # vis_dir = 'Case_study/{}'.format(IS.dataset_name)
    # os.makedirs(vis_dir, exist_ok=True)
    # for line in tqdm(lines):
    #     pair_ind1, pair_ind2 = line.strip().split(',')
    #     pair_ind1, pair_ind2 = int(pair_ind1), int(pair_ind2)
    #     os.makedirs('./{}/{}_{}/'.format(vis_dir, pair_ind1, pair_ind2), exist_ok=True)
    #     with open('./{}/Allrelabeldict_{}_{}_soft_knn.pkl'.format(base_dir, pair_ind1, pair_ind2), 'rb') as handle:
    #         relabel_dict = pickle.load(handle)
    #
    #     for ind in relabel_dict.keys():
    #         orig_cls = IS.dl_tr.dataset.ys[ind]
    #         recommend_cls = relabel_dict[ind].argsort()[::-1][0]
    #         if recommend_cls != orig_cls:
    #             fig = plt.figure(figsize=(30, 15))
    #             plt.subplot(2, 6, 1)
    #             img = read_image(IS.dl_tr.dataset.im_paths[ind])
    #             plt.imshow(to_pil_image(img))
    #             plt.axis('off')
    #             if lookat_harmful:
    #                 plt.title('Harmful Training', fontdict={'fontsize': 20})
    #             else:
    #                 plt.title('Helpful Training', fontdict={'fontsize': 20})
    #
    #             # sample images
    #             orig_cls_indices = np.where(np.asarray(IS.dl_tr.dataset.ys) == orig_cls)[0]
    #             recom_cls_indices = np.where(np.asarray(IS.dl_tr.dataset.ys) == recommend_cls)[0]
    #             for i in range(min(5, len(orig_cls_indices))):
    #                 plt.subplot(2, 6, i + 2)
    #                 img = read_image(IS.dl_tr.dataset.im_paths[orig_cls_indices[i]])
    #                 plt.imshow(to_pil_image(img))
    #                 plt.axis('off')
    #                 if i == 2:
    #                     plt.title('Original Class = {}'.format(orig_cls), fontdict={'fontsize': 20})
    #
    #             for i in range(min(5, len(recom_cls_indices))):
    #                 plt.subplot(2, 6, i + 3 + 5)
    #                 img = read_image(IS.dl_tr.dataset.im_paths[recom_cls_indices[i]])
    #                 plt.imshow(to_pil_image(img))
    #                 plt.axis('off')
    #                 if i == 2:
    #                     plt.title('Recommend Class = {}'.format(recommend_cls), fontdict={'fontsize': 20})
    #             plt.tight_layout()
    #             # plt.show()
    #             plt.savefig('./{}/{}_{}/harmful{}_{}.png'.format(vis_dir, pair_ind1, pair_ind2, lookat_harmful, ind))
    #
    #     break

    '''Step 4: Train with relabelled data'''
    for line in tqdm(lines):
        torch.cuda.empty_cache()
        pair_ind1, pair_ind2 = line.strip().split(',')
        pair_ind1, pair_ind2 = int(pair_ind1), int(pair_ind2)
        #  training with relabelled data
        os.system("python train_sample_relabel.py --dataset {} \
                    --loss-type ProxyNCA_prob_orig_{}_relabel_{}_{} \
                    --relabel_dict Confuse_pair_influential_data/{}/Allrelabeldict_{}_{}_{}.pkl \
                    --model_dir {} \
                    --seed {} --config config/{}_relabel.json".format(IS.dataset_name,
                                                                       relabel_method,
                                                                       pair_ind1, pair_ind2,
                                                                       IS.dataset_name,
                                                                       pair_ind1, pair_ind2, relabel_method,
                                                                       IS.model_dir,
                                                                       IS.seed,
                                                                       IS.dataset_name))
    exit()

    '''Step 5: Verify that the model after training is better?'''
    new_features = IS.get_test_features()
    _, recall, _ = evaluate(IS.model, IS.dl_ev, eval_nmi=False)
    print(recall)
    for line in lines:
        pair_ind1, pair_ind2 = line.strip().split(',')
        pair_ind1, pair_ind2 = int(pair_ind1), int(pair_ind2)
        print('For pair: ', pair_ind1, pair_ind2)
        IS.model = IS._load_model()  # reload the original weights
        inter_dist_orig, _ = grad_confusion_pair(IS.model, new_features, [pair_ind1], [pair_ind2])
        print('Original distance: ', inter_dist_orig)

        new_weight_path = 'models/dvi_data_{}_{}_loss{}/ResNet_512_Model/Epoch_{}/{}_{}_trainval_{}_{}.pth'.format(
                            dataset_name,
                            seed,
                            'ProxyNCA_prob_orig_{}_relabel_{}_{}'.format(relabel_method, pair_ind1, pair_ind2),
                            5, dataset_name,
                            dataset_name,
                            512, seed)  # reload weights as new
        IS.model.load_state_dict(torch.load(new_weight_path))
        _, recall, _ = evaluate(IS.model, IS.dl_ev, eval_nmi=False)
        print('After recall:', recall)
        inter_dist_after, _ = grad_confusion_pair(IS.model, new_features, [pair_ind1], [pair_ind2])
        print('After distance: ', inter_dist_after)
