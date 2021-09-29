import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import utils
import dataset
from tqdm import tqdm
from networks import bninception, Feat_resnet50_max_n
from torch import nn
import umap
from umap.parametric_umap import ParametricUMAP
import tensorflow as tf
from loss import ProxyNCA_prob, ProxyNCA_prob_orig
from utils import predict_batchwise, inner_product_sim, predict_batchwise_loss
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
from visualize.parametric_umap import prepare_data, encoder_model, encoder_trainer

def pumap_training_all(model, model_dir, e,
                   dl_tr, dl_ev,
                   stacked_proxies, presaved, pretrained,
                       num_class_show):
    '''
        Parameteric Umap training
    '''
    if not presaved:
        train_embedding, train_label, *_ = predict_batchwise(model, dl_tr)
        torch.save(train_embedding, '{}/Epoch_{}/training_embeddings.pth'.format(model_dir, e + 1))
        torch.save(train_label, '{}/Epoch_{}/training_labels.pth'.format(model_dir, e + 1))

    else:
        train_embedding = torch.load('{}/Epoch_{}/training_embeddings.pth'.format(model_dir, e + 1))
        train_label = torch.load('{}/Epoch_{}/training_labels.pth'.format(model_dir, e + 1))

    torch.cuda.empty_cache()
    if not presaved:
        test_embedding, test_label, *_ = predict_batchwise(model, dl_ev)
        torch.save(test_embedding, '{}/Epoch_{}/testing_embeddings.pth'.format(model_dir, e + 1))
        torch.save(test_label, '{}/Epoch_{}/testing_labels.pth'.format(model_dir, e + 1))
    else:
        test_embedding = torch.load('{}/Epoch_{}/testing_embeddings.pth'.format(model_dir, e + 1))
        test_label = torch.load('{}/Epoch_{}/testing_labels.pth'.format(model_dir, e + 1))

    # need to normalize, other producing wierd results
    train_embedding = F.normalize(train_embedding, dim=-1)
    test_embedding = F.normalize(test_embedding, dim=-1)
    stacked_proxies = F.normalize(stacked_proxies, dim=-1)
    print('Training Embedding of shape: ', train_embedding.shape,
          'Testing Embedding of shape: ', test_embedding.shape,
          'Current proxies of shape: ', stacked_proxies.shape)

    # FIXME: fit on subset of data only
    train_embedding = train_embedding[train_label < num_class_show]
    train_label = train_label[train_label < num_class_show]

    test_embedding = test_embedding[test_label < torch.min(test_label).item()+50]
    test_label = test_label[test_label < torch.min(test_label).item()+50]
    print('Training Embedding of shape: ', train_embedding.shape,
          'Testing Embedding of shape: ', test_embedding.shape,
          'Current proxies of shape: ', stacked_proxies.shape)

    # concatenate all
    all_data = np.concatenate((train_embedding.detach().cpu().numpy(),
                               test_embedding.detach().cpu().numpy(),
                               stacked_proxies.cpu().numpy()), axis=0)

    # Parametric Umap model
    embedder = encoder_trainer(model_dir, all_data, pretrained, e)

    return embedder, train_embedding, train_label, test_embedding, test_label

def visualize_interactive_all(low_dim_train_emb, low_dim_proxy_emb, low_dim_test_emb,
                              train_label, proxy_label, test_label, interactive=True):
    # # Visualize
    fig, ax = plt.subplots(figsize=(20, 20))
    # For embedding points
    tr_x, tr_y = low_dim_train_emb[:, 0], low_dim_train_emb[:, 1]
    te_x, te_y = low_dim_test_emb[:, 0], low_dim_test_emb[:, 1]
    px, py = low_dim_proxy_emb[:, 0], low_dim_proxy_emb[:, 1]

    scatter4tr = ax.scatter(tr_x, tr_y, c='gray', s=10, zorder=1)
    scatter4te = ax.scatter(te_x, te_y, c='pink', s=10, zorder=2)
    scatter4proxy = ax.scatter(px, py, c='blue', s=20, marker='*', zorder=3)

    if interactive:
        sample_plots = []
        plot = ax.plot([], [], '.', ms=10, color='darkblue', zorder=4)
        sample_plots.append(plot[0])
        plot[0].set_visible(False)

        sample_plots2 = []
        plot2 = ax.plot([], [], '.', ms=10, color='red', zorder=5)
        sample_plots2.append(plot2[0])
        plot2[0].set_visible(False)

        sample_plots3 = []
        plot3 = ax.plot([], [], '*', ms=15, color='yellow', zorder=6)
        sample_plots3.append(plot3[0])
        plot3[0].set_visible(False)

        def hover(event):
            # if the mouse is over the scatter points
            if scatter4tr.contains(event)[0]:
                # find out the index within the array from the event
                ind = scatter4tr.contains(event)[1]["ind"]
                c = train_label[ind]
                data = low_dim_train_emb[train_label == c, :]
                plot[0].set_visible(True)
                sample_plots[0].set_data(data.transpose())
            else:
                # if the mouse is not over a scatter point
                plot[0].set_visible(False)
                fig.canvas.draw_idle()

            if scatter4proxy.contains(event)[0]:
                ind = scatter4proxy.contains(event)[1]["ind"]
                c = proxy_label[ind]
                proxy_data = low_dim_proxy_emb[proxy_label == c, :]
                plot3[0].set_visible(True)
                sample_plots3[0].set_data(proxy_data.transpose())
            else:
                plot3[0].set_visible(False)
                fig.canvas.draw_idle()

            # if the mouse is over the scatter points
            if scatter4te.contains(event)[0]:
                # find out the index within the array from the event
                ind = scatter4te.contains(event)[1]["ind"]
                c = test_label[ind]
                data = low_dim_test_emb[test_label == c, :]
                plot2[0].set_visible(True)
                sample_plots2[0].set_data(data.transpose())
            else:
                # if the mouse is not over a scatter point
                plot2[0].set_visible(False)
                fig.canvas.draw_idle()


        # # add callback for mouse moves
        fig.canvas.mpl_connect('motion_notify_event', hover)
    plt.draw()
    plt.show()



if __name__ == '__main__':
    dataset_name = 'logo2k_super100'
    config_name = 'logo2k_orig'
    dynamic_proxy = False
    sz_embedding = 2048
    tau = 0.0
    presaved = True
    pretrained = True
    initial_proxy_num = 1
    interactive = False
    num_class_show = 10

    folder = 'dvi_data_{}_{}_t0.1_proxy{}_tau{}/'.format(dataset_name, dynamic_proxy, initial_proxy_num, tau)
    model_dir = '{}/ResNet_{}_Model'.format(folder, sz_embedding)
    plot_dir = '{}/resnet_{}_umap_plots'.format(folder, sz_embedding)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, 'Training_data'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'Testing_data'), exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    dl_tr, dl_ev = prepare_data(data_name=dataset_name,
                                config_file=config_name,
                                root=folder,
                                save=False)

    # load model
    feat = Feat_resnet50_max_n()
    in_sz = feat(torch.rand(1, 3, 256, 256)).squeeze().size(0)
    feat.train()
    emb = torch.nn.Linear(in_sz, sz_embedding)
    model = torch.nn.Sequential(feat, emb)
    model = nn.DataParallel(model)
    model.cuda()

    # load loss
    # FIXME: should save the whole criterion class because mask might be updating
    # criterion = ProxyNCA_prob(nb_classes = dl_tr.dataset.nb_classes(),
    #                           sz_embed = sz_embedding,
    #                           scale=3,
    #                           initial_proxy_num=initial_proxy_num)
    criterion = ProxyNCA_prob_orig(nb_classes = dl_tr.dataset.nb_classes(),
                              sz_embed = sz_embedding,
                              scale=3)

    for i in range(1, 2):
        subclasses = np.asarray(list(range(5*(i-1), 5*i)))
        for e in tqdm([39]):
            model.load_state_dict(torch.load('{}/Epoch_{}/{}_{}_trainval_{}_0.pth'.format(model_dir, e+1, dataset_name,
                                                                                          dataset_name, sz_embedding)))
            proxies = torch.load('{}/Epoch_{}/proxy.pth'.format(model_dir, e+1), map_location='cpu')['proxies'].detach()
            # reshape_proxies = proxies.view(criterion.nb_classes,
            #                                criterion.max_proxy_per_class, -1)
            # mask = torch.load('{}/Epoch_{}/proxy.pth'.format(model_dir, e+1), map_location='cpu')['mask'].detach()
            # count_proxy = torch.sum(mask, -1).detach().cpu().numpy().tolist()

            # used_proxies = []
            # for m, n in enumerate(count_proxy):
            #     used_proxies.append(reshape_proxies[m, :int(n)]) # of shape (C, sz_embedding)
            # stacked_proxies = torch.cat(used_proxies, dim=0)

            # TODO: delete
            count_proxy = [1]*dl_tr.dataset.nb_classes()
            stacked_proxies = proxies
            used_proxies = [x for x in proxies]

            #TODO: reload criterion
            criterion.proxies.data = proxies
            # criterion.mask = mask
            embedder, train_embedding, train_label, test_embedding, test_label = pumap_training_all(model, model_dir, e,
                                                                                                   dl_tr, dl_ev,
                                                                                                   stacked_proxies, presaved, pretrained,
                                                                                                    num_class_show)

            # transform high dimensional embedding and proxy to low-dimension
            low_dim_train_emb = embedder.transform(train_embedding.detach().cpu().numpy())
            low_dim_test_emb = embedder.transform(test_embedding.detach().cpu().numpy())
            # low_dim_proxy_emb = []
            # for p in used_proxies:
            #     p = F.normalize(p, p=2, dim=-1)
            #     low_dim_proxy_emb.append(embedder.transform(p.cpu().numpy()))
            # TODO delete
            low_dim_proxy_emb = embedder.transform(F.normalize(stacked_proxies, p=2, dim=-1).cpu().numpy())

            #
            # low_dim_proxy_sub = []
            # low_dim_proxy_labels = []
            # for m, p in enumerate(low_dim_proxy_emb):
            #     if m < num_class_show:
            #         for sub_p in p:
            #             low_dim_proxy_labels.append(m)
            #             low_dim_proxy_sub.append(sub_p)

            # low_dim_proxy_sub = np.asarray(low_dim_proxy_sub)
            # low_dim_proxy_labels = np.asarray(low_dim_proxy_labels)
            # print(low_dim_proxy_sub.shape)

            # TODO delete
            low_dim_proxy_labels = []
            low_dim_proxy_sub = []
            for m, p in enumerate(low_dim_proxy_emb):
                if m < num_class_show:
                    low_dim_proxy_labels.append(m)
                    low_dim_proxy_sub.append(p)
            low_dim_proxy_labels = np.asarray(low_dim_proxy_labels)
            low_dim_proxy_sub = np.asarray(low_dim_proxy_sub)

            visualize_interactive_all(low_dim_train_emb, low_dim_proxy_sub, low_dim_test_emb,
                                      train_label, low_dim_proxy_labels, test_label, True)