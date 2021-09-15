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
from umap.parametric_umap import load_ParametricUMAP
from loss import ProxyNCA_prob
from utils import predict_batchwise, inner_product_sim
import json
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = "1,0"


def prepare_data(data_name='cub', root='dvi_data_cub200/', save=False):
    '''
    Prepare dataloader
    :param data_name: dataset used
    :param root:root dir to save data
    :param save: if save is True, save data
    '''
    dataset_config = utils.load_config('dataset/config.json')

    config = utils.load_config('config/{}.json'.format(data_name))
    transform_key = 'transform_parameters'
    if 'transform_key' in config.keys():
        transform_key = config['transform_key']
    print('Transformation: ', transform_key)

    dl_tr_noshuffle = torch.utils.data.DataLoader(
            dataset=dataset.load(
                    name=data_name,
                    root=dataset_config['dataset'][data_name]['root'],
                    source=dataset_config['dataset'][data_name]['source'],
                    classes=dataset_config['dataset'][data_name]['classes']['trainval'],
                    transform=dataset.utils.make_transform(
                        **dataset_config[transform_key],
                        is_train=False
                    )
                ),
            num_workers = 0,
            shuffle=False,
            batch_size=64,
    )

    dl_ev = torch.utils.data.DataLoader(
        dataset.load(
            name=data_name,
            root=dataset_config['dataset'][data_name]['root'],
            source=dataset_config['dataset'][data_name]['source'],
            classes=dataset_config['dataset'][data_name]['classes']['eval'],
            transform=dataset.utils.make_transform(
                **dataset_config[transform_key],
                is_train=False
            )
        ),
        batch_size=128,
        shuffle=False,
        num_workers=0,

    )

    if save:
        training_x, training_y = torch.tensor([]), torch.tensor([])
        for ct, (x,y,_) in tqdm(enumerate(dl_tr_noshuffle)): #FIXME: memory error
            training_x = torch.cat((training_x, x), dim=0)
            training_y = torch.cat((training_y, y), dim=0)
        torch.save(training_x, os.path.join(root, 'Training_data', 'training_dataset_data.pth'))
        torch.save(training_y, os.path.join(root, 'Training_data', 'training_dataset_label.pth'))

        test_x, test_y = torch.tensor([]), torch.tensor([])
        for ct, (x,y,_) in tqdm(enumerate(dl_ev)):
            test_x = torch.cat((test_x, x), dim=0)
            test_y = torch.cat((test_y, y), dim=0)
        torch.save(test_x, os.path.join(root, 'Testing_data', 'testing_dataset_data.pth'))
        torch.save(test_y, os.path.join(root, 'Testing_data', 'testing_dataset_label.pth'))

    return dl_tr_noshuffle, dl_ev


def encoder_model(n_components=2):
    '''
    Customized encoder
    :param n_components: low dimensional projection dimensions
    '''
    encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(units=256, activation="relu"),
            tf.keras.layers.Dense(units=256, activation="relu"),
            tf.keras.layers.Dense(units=n_components),
        ])
    return encoder

if __name__ == '__main__':

    dataset_name = 'logo2k'
    dynamic_proxy = True
    os.makedirs('dvi_data_{}_{}/'.format(dataset_name, dynamic_proxy), exist_ok=True)
    os.makedirs(os.path.join('dvi_data_{}_{}/'.format(dataset_name, dynamic_proxy), 'Training_data'), exist_ok=True)
    os.makedirs(os.path.join('dvi_data_{}_{}/'.format(dataset_name, dynamic_proxy), 'Testing_data'), exist_ok=True)
    os.makedirs('dvi_data_{}_{}/resnet_2048_umap_plots/'.format(dataset_name, dynamic_proxy), exist_ok=True)

    model_dir = 'dvi_data_{}_{}/ResNet_2048_Model'.format(dataset_name, dynamic_proxy)
    plot_dir = 'dvi_data_{}_{}/resnet_2048_umap_plots'.format(dataset_name, dynamic_proxy)
    sz_embedding = 2048

    os.makedirs(plot_dir, exist_ok=True)
    dl_tr, dl_ev = prepare_data(data_name=dataset_name, root='dvi_data_{}_{}/'.format(dataset_name, dynamic_proxy), save=False)

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
    criterion = ProxyNCA_prob(
        nb_classes = dl_tr.dataset.nb_classes(),
        sz_embed = sz_embedding,
        scale=3)

    with open("{0}/{1}_ip.json".format('log', '{}_{}_trainval_2048_0_{}'.format(dataset_name, dataset_name, dynamic_proxy)), 'rt') as handle:
        cache_sim = json.load(handle)
    with open("{0}/{1}_cls.json".format('log', '{}_{}_trainval_2048_0_{}'.format(dataset_name, dataset_name, dynamic_proxy)), 'rt') as handle:
        cache_label = json.load(handle)

    # Line plot which show the trend of inner_prod_sim to nearest ground-truth class's proxy
    os.makedirs(os.path.join(plot_dir, 'line_plot'), exist_ok=True)
    for cls in range(criterion.nb_classes):
        sim_cls1 = []
        for e in tqdm(range(40)):
            sim = cache_sim[str(e)]
            labels = cache_label[str(e)]
            sim_cls = np.asarray(sim)[np.asarray(labels) == cls].tolist()
            sim_cls1.append(sim_cls)

        sim_cls1 = np.asarray(sim_cls1)
        fig = plt.figure()
        for j in range(sim_cls1.shape[1]):
            plt.plot(range(40), sim_cls1[:, j], linestyle='-', color='k', linewidth=0.5)
        fig.savefig(os.path.join(plot_dir, 'line_plot', 'cls{}.png'.format(str(cls))))

    # for i in range(117, 118):
    #     subclasses = np.asarray(list(range(10*(i-1), 10*i)))
    #     for e in tqdm([0, 9, 19, 29, 39]):
    #
    #         model.load_state_dict(torch.load('{}/Epoch_{}/{}_{}_trainval_2048_0.pth'.format(model_dir, e+1, dataset_name, dataset_name)))
    #         proxies = torch.load('{}/Epoch_{}/proxy.pth'.format(model_dir, e+1), map_location='cpu')['proxies'].detach()
    #         proxies = proxies.view(criterion.nb_classes, criterion.max_proxy_per_class, -1)
    #         # FIXME: get only the 1st proxy each class
    #         used_proxies = proxies[:, 0, :] # of shape (C, sz_embedding)
    #
    #         # embedding, label, *_ = predict_batchwise(model, dl_tr)
    #         # torch.save(embedding, '{}/Epoch_{}/training_embeddings.pth'.format(model_dir, e+1))
    #         # torch.save(label, '{}/Epoch_{}/training_labels.pth'.format(model_dir, e+1))
    #         embedding = torch.load('{}/Epoch_{}/training_embeddings.pth'.format(model_dir, e+1))
    #         label = torch.load('{}/Epoch_{}/training_labels.pth'.format(model_dir, e+1))
    #         embedding = F.normalize(embedding, dim=-1) # normalize?
    #         used_proxies = F.normalize(used_proxies, dim=-1)
    #         print(embedding.shape)
    #         print(used_proxies.shape)
    #
    #         # Parametric Umap model
    #         encoder = encoder_model()
    #         embedder = ParametricUMAP(encoder=encoder, verbose=False, batch_size=256)
    #
    #         if e > 0:
    #             # Initialize by last visualization model
    #             embedder.encoder = tf.keras.models.load_model('{}/Epoch_{}/parametric_model/encoder'.format(model_dir, e))
    #         # Train on all samples and all proxies
    #         # embedder.fit_transform(np.concatenate((embedding.detach().cpu().numpy(), used_proxies.cpu().numpy()), axis=0))
    #         # embedder.encoder.save('{}/Epoch_{}/parametric_model/encoder'.format(model_dir, e+1))
    #         embedder.encoder = tf.keras.models.load_model('{}/Epoch_{}/parametric_model/encoder'.format(model_dir, e+1))
    #
    #         low_dim_emb = embedder.transform(embedding.detach().cpu().numpy())
    #         low_dim_proxy = embedder.transform(used_proxies.cpu().numpy())
    #         print(low_dim_emb.shape)
    #         print(low_dim_proxy.shape)
    #
    #         # Only visualize first 10 classes
    #         label_sub = label[np.isin(label, subclasses)].numpy()
    #         low_dim_emb = low_dim_emb[np.isin(label, subclasses), :]
    #         low_dim_proxy = low_dim_proxy[subclasses]
    #
    #         # Visualize
    #         fig = plt.figure()
    #         scatter = plt.scatter(low_dim_emb[:, 0], low_dim_emb[:, 1], c=label_sub, cmap='tab10', s=5)
    #         classes = subclasses.tolist()
    #         plt.scatter(low_dim_proxy[:, 0], low_dim_proxy[:, 1], c=range(10), cmap='tab10', marker=(5,1), edgecolors='black')
    #         plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    #         os.makedirs('{}/{}th_batch'.format(plot_dir, str(i)), exist_ok=True)
    #         fig.savefig('{}/{}th_batch/Epoch_{}.png'.format(plot_dir, str(i), e+1))