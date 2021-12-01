import numpy as np
import torch
import torchvision
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

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == '__main__':

    dataset_name = 'sop'
    loss_type = 'ProxyNCA_pfix_EM'
    config_name = 'cars'
    sz_embedding = 512
    seed = 1


    folder = 'dvi_data_{}_{}_loss{}/'.format(dataset_name, seed, loss_type)
    model_dir = '{}/ResNet_{}_Model'.format(folder, sz_embedding)
    plot_dir = '{}/resnet_{}_umap_plots'.format(folder, sz_embedding)

    # load data
    dl_tr, dl_ev = prepare_data(data_name=dataset_name, config_name=config_name,
                                root=folder, save=False)

    # load model
    feat = Feat_resnet50_max_n()
    in_sz = feat(torch.rand(1, 3, 256, 256)).squeeze().size(0)
    feat.train()
    emb = torch.nn.Linear(in_sz, sz_embedding)
    model = torch.nn.Sequential(feat, emb)
    model = nn.DataParallel(model)
    model.cuda()

    model.load_state_dict(
        torch.load('{}/Epoch_{}/{}_{}_trainval_512_{}.pth'.format(model_dir, 40, dataset_name, dataset_name, seed)))
    proxies = torch.load('{}/Epoch_{}/proxy.pth'.format(model_dir, 40), map_location='cpu')['proxies'].detach()

    # load pre-saved
    train_embedding = torch.load(
        '{}/Epoch_{}/training_embeddings.pth'.format(model_dir, 40))  # high dimensional embedding
    train_label = torch.load('{}/Epoch_{}/training_labels.pth'.format(model_dir, 40))
    testing_embedding = torch.load(
        '{}/Epoch_{}/testing_embeddings.pth'.format(model_dir, 40))  # high dimensional embedding
    testing_label = torch.load('{}/Epoch_{}/testing_labels.pth'.format(model_dir, 40))

    '''Get top 15 most wrong classes (TEST)'''
    wrong_ind, top15_wrong_classes = get_wrong_indices(testing_embedding, testing_label)

    '''get train classes that are closest to these 15 test classes'''
    # 1st wrong class
    selected_test_class = top15_wrong_classes[0]
    print('Investigating testing class: ', selected_test_class)
    selected_test_embed = testing_embedding[testing_label == selected_test_class] # (N_ev, sz_embed)
    print(selected_test_embed.shape)

    # get closest train classes to the selected test class
    test2trainD = torch.zeros(dl_tr.dataset.nb_classes())
    for cls in range(dl_tr.dataset.nb_classes()):
        indices = train_label == cls
        selected_train_embed = train_embedding[indices] # (N_tr, sz_embed)
        D = utils.pairwise_distance(
            torch.cat([selected_test_embed,
                       selected_train_embed]),
            squared=False
        )[0][:len(selected_test_embed), len(selected_test_embed):] # (N_ev, N_tr)
        row_ind, col_ind = linear_sum_assignment(D.detach().cpu().numpy())
        bestD = D[row_ind, col_ind].sum()
        test2trainD[cls] = bestD

    topk_closest_traincls = np.arange(dl_tr.dataset.nb_classes())[np.argsort(test2trainD)][:1] # closeset classes
    print('Topk closeset training class', topk_closest_traincls)

    '''sanity check'''
    # import matplotlib.pyplot as plt
    # indices = np.where(testing_label == 178)[0]
    # check = dl_ev.dataset.__getitem__(indices[0])
    # plt.imshow(check[0].permute(1, 2, 0)[:, :, [2,1,0]])
    # plt.show()
    #
    # indices = np.where(train_label == 17)[0]
    # plt.imshow(dl_tr.dataset.__getitem__(indices[0])[0].permute(1, 2, 0)[:, :, [2,1,0]])
    # plt.show()
    #
    # indices = np.where(train_label == 5)[0]
    # plt.imshow(dl_tr.dataset.__getitem__(indices[0])[0].permute(1, 2, 0)[:, :, [2,1,0]])
    # plt.show()
    #
    # indices = np.where(train_label == 4)[0]
    # plt.imshow(dl_tr.dataset.__getitem__(indices[0])[0].permute(1, 2, 0)[:, :, [2,1,0]])
    # plt.show()

    '''Use PCA to get 1st PC, project training samples onto the 1st PC and identify which samples are singificantly contributing'''
    # pca = PCA(n_components=1)
    # projected_test_emb = pca.fit_transform(selected_test_embed.detach().cpu().numpy()) # (N_test, 1)
    # projected_train_emb = pca.transform(train_embedding.detach().cpu().numpy()) # (N_train, 1)
    # sig_contributing_train_indices = np.argsort(projected_train_emb[:, 0])[::-1]
    #
    # w = 10; h = 10; columns = 3; rows = 3
    # fig = plt.figure(figsize=(8, 8))
    # for i in range(1, columns * rows + 1):
    #     img = np.random.randint(10, size=(h, w))
    #     fig.add_subplot(rows, columns, i)
    #     img = read_image(dl_tr.dataset.im_paths[sig_contributing_train_indices[i]])
    #     plt.imshow(to_pil_image(img))
    #     plt.title(dl_tr.dataset.__getitem__(sig_contributing_train_indices[i])[1])
    # plt.show()

    '''Use CAM methods to retrieve the contributing pixels'''
    model_full = Full_Model(emb_size=sz_embedding, num_classes=dl_tr.dataset.nb_classes())
    pretrained_feat = torch.load('{}/Epoch_{}/{}_{}_trainval_512_{}.pth'.format(model_dir, 40, dataset_name, dataset_name, seed))
    model_full.model.load_state_dict({k.replace('module.', ''):v for k,v in pretrained_feat.items()})
    model_full.proj.weight.data = F.normalize(proxies, p=2, dim=-1) * 2 * 9 # T = 1/9
    model_full.proj.bias.data = torch.ones(dl_tr.dataset.nb_classes()) * -2 * 9 # T = 1/9
    model_full.eval()
    model_full.cuda()

    cam_extractor = SmoothGradCAMpp(model_full)
    dl_tr, dl_ev = prepare_data(data_name=dataset_name, config_name=config_name,
                                root=folder, batch_size=1, save=False)

    for ct, (x, y, indices) in tqdm(enumerate(dl_tr)):
        if y.item() in topk_closest_traincls:
            os.makedirs('CAM', exist_ok=True)
            os.makedirs('CAM/{}_{}/testcls{}_traincls{}_trainCAM'.format(dataset_name, loss_type, selected_test_class, y.item()), exist_ok=True)

            x, y = x.cuda(), y.cuda()
            out = model_full(x)
            # Retrieve the CAM by passing the class index and the model output
            activation_map = cam_extractor(y.item(), out)
            # Resize the CAM and overlay it
            img = read_image(dl_tr.dataset.im_paths[indices[0]])
            result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].detach().cpu(), mode='F'), alpha=0.5)
            # Display it
            plt.imshow(result); plt.axis('off'); plt.tight_layout()
            plt.savefig(os.path.join('CAM/{}_{}/testcls{}_traincls{}_trainCAM'.format(dataset_name, loss_type, selected_test_class, y.item()),
                                     os.path.basename(dl_tr.dataset.im_paths[indices[0]])))

    for ct, (x, y, indices) in tqdm(enumerate(dl_ev)):
        if y.item() in [selected_test_class] and indices.item() in wrong_ind:
            os.makedirs('CAM', exist_ok=True)
            os.makedirs('CAM/{}_{}/testcls{}_traincls{}_testCAM'.format(dataset_name, loss_type, y.item(), topk_closest_traincls[0]), exist_ok=True)
            x, y = x.cuda(), y.cuda()
            out = model_full(x)
            # Retrieve the CAM by passing the class index and the model output
            activation_map = cam_extractor(topk_closest_traincls[0].item(), out)
            # Resize the CAM and overlay it
            img = read_image(dl_ev.dataset.im_paths[indices[0]])
            result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].detach().cpu(), mode='F'), alpha=0.5)
            # Display it
            plt.imshow(result); plt.axis('off'); plt.tight_layout()
            plt.savefig(os.path.join('CAM/{}_{}/testcls{}_traincls{}_testCAM'.format(dataset_name, loss_type, y.item(), topk_closest_traincls[0]),
                                     os.path.basename(dl_ev.dataset.im_paths[indices[0]])))





