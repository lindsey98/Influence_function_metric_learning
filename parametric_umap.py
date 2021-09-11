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
from utils import predict_batchwise
os.environ['CUDA_VISIBLE_DEVICES'] = "1,0"

os.makedirs('dvi_data_logo2k/', exist_ok=True)
os.makedirs(os.path.join('dvi_data_logo2k/', 'Training_data'), exist_ok=True)
os.makedirs(os.path.join('dvi_data_logo2k/', 'Testing_data'), exist_ok=True)
os.makedirs(os.path.join('dvi_data_logo2k/', 'Training_data_resnet'), exist_ok=True)
os.makedirs(os.path.join('dvi_data_logo2k/', 'Testing_data_resnet'), exist_ok=True)
os.makedirs('dvi_data_logo2k/umap_plots/', exist_ok=True)
os.makedirs('dvi_data_logo2k/resnet_2048_umap_plots/', exist_ok=True)

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
                    classes=dataset_config['dataset'][data_name]['classes']['train'],
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
        torch.save(training_x, os.path.join(root, 'Training_data_resnet', 'training_dataset_data.pth'))
        torch.save(training_y, os.path.join(root, 'Training_data_resnet', 'training_dataset_label.pth'))

        test_x, test_y = torch.tensor([]), torch.tensor([])
        for ct, (x,y,_) in tqdm(enumerate(dl_ev)):
            test_x = torch.cat((test_x, x), dim=0)
            test_y = torch.cat((test_y, y), dim=0)
        torch.save(test_x, os.path.join(root, 'Testing_data_resnet', 'testing_dataset_data.pth'))
        torch.save(test_y, os.path.join(root, 'Testing_data_resnet', 'testing_dataset_label.pth'))

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
    model_dir = 'dvi_data_logo2k/ResNet_2048_Model'
    plot_dir = 'dvi_data_logo2k/resnet_2048_umap_plots'
    sz_embedding = 2048

    os.makedirs(plot_dir, exist_ok=True)
    dl_tr, dl_ev = prepare_data(data_name='logo2k', root='dvi_data_logo2k/', save=False)

    # load model
    feat = Feat_resnet50_max_n()
    in_sz = feat(torch.rand(1, 3, 256, 256)).squeeze().size(0)
    feat.train()
    emb = torch.nn.Linear(in_sz, sz_embedding)
    model = torch.nn.Sequential(feat, emb)
    model = nn.DataParallel(model)
    model.cuda()

    for e in tqdm(range(40)):

        model.load_state_dict(torch.load('{}/Epoch_{}/logo2k_logo2k_trainval_2048_0.pth'.format(model_dir, e+1)))
        proxies = torch.load('{}/Epoch_{}/proxy.pth'.format(model_dir, e+1), map_location='cpu').detach().numpy()

        embedding, label, *_ = predict_batchwise(model, dl_tr)
        torch.save(embedding, '{}/Epoch_{}/training_embeddings.pth'.format(model_dir, e+1))
        torch.save(label, '{}/Epoch_{}/training_labels.pth'.format(model_dir, e+1))

        # Parametric Umap model
        encoder = encoder_model()
        embedder = ParametricUMAP(encoder=encoder, verbose=False)

        if e > 0:
            # Initialize by last visualization model
            embedder.encoder = tf.keras.models.load_model('{}/Epoch_{}/parametric_model/encoder'.format(model_dir, e))
        # Train on all samples and all proxies
        embedder.fit_transform(np.concatenate((embedding.numpy(), proxies), axis=0))
        embedder.encoder.save('{}/Epoch_{}/parametric_model/encoder'.format(model_dir, e+1))

        # Only visualize first 10 classes
        embedding_sub = embedding[label < 10, :].numpy()
        label_sub = label[label < 10].numpy()
        low_dim_emb = embedder.transform(embedding_sub)
        low_dim_proxy = embedder.transform(proxies[:10])

        # Visualize
        fig = plt.figure()
        plt.scatter(low_dim_emb[:, 0], low_dim_emb[:, 1], c=label_sub, cmap='tab10', s=5)
        plt.legend()
        plt.scatter(low_dim_proxy[:, 0], low_dim_proxy[:, 1], c=range(10), cmap='tab10', marker=(5,1))
        fig.savefig('{}/Epoch_{}.png'.format(plot_dir, e+1))

