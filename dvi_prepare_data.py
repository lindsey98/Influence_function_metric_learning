import matplotlib.pyplot as plt
import torch
import os
import utils
import dataset
from tqdm import tqdm
from networks import bninception
from torch import nn
import umap
from umap.parametric_umap import ParametricUMAP
import tensorflow as tf
from umap.parametric_umap import load_ParametricUMAP
from loss import ProxyNCA_prob

os.makedirs('dvi_data/', exist_ok=True)
os.makedirs(os.path.join('dvi_data/', 'Training_data'), exist_ok=True)
os.makedirs(os.path.join('dvi_data/', 'Testing_data'), exist_ok=True)
os.makedirs(os.path.join('dvi_data/', 'Model'), exist_ok=True)
os.makedirs('dvi_data/umap_plots/', exist_ok=True)

def prepare_data(data_name='cub'):
    ################ Prepare data ################
    dataset_config = utils.load_config('dataset/config.json')
    # dataset_config['dataset'][data_name]['root'] = 'mnt/datasets/CUB_200_2011'

    config = utils.load_config('config/{}.json'.format(data_name))
    transform_key = 'transform_parameters'
    if 'transform_key' in config.keys():
        transform_key = config['transform_key']
    print('Transformation: ', transform_key)

    train_transform = dataset.utils.make_transform(
        **dataset_config[transform_key],
         is_train = False # disable fancy transformations
    )
    tr_dataset = dataset.load(
        name=data_name,
        root=dataset_config['dataset'][data_name]['root'],
        source=dataset_config['dataset'][data_name]['source'],
        classes=dataset_config['dataset'][data_name]['classes']['trainval'],
        transform=train_transform
    )


    print(len(tr_dataset))

    dl_tr = torch.utils.data.DataLoader(
        tr_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
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
        num_workers=8,
    )

    # training_x = torch.tensor([])
    # training_y = torch.tensor([])
    #
    # for ct, (x,y,_) in tqdm(enumerate(dl_tr)):
    #     training_x = torch.cat((training_x, x), dim=0)
    #     training_y = torch.cat((training_y, y), dim=0)
    #
    # torch.save(training_x, os.path.join('dvi_data/', 'Training_data', 'training_dataset_data.pth'))
    # torch.save(training_y, os.path.join('dvi_data/', 'Training_data', 'training_dataset_label.pth'))

    # test_x = torch.tensor([])
    # test_y = torch.tensor([])
    #
    # for ct, (x,y,_) in tqdm(enumerate(dl_ev)):
    #     test_x = torch.cat((test_x, x), dim=0)
    #     test_y = torch.cat((test_y, y), dim=0)
    #
    # torch.save(test_x, os.path.join('dvi_data/', 'Testing_data', 'testing_dataset_data.pth'))
    # torch.save(test_y, os.path.join('dvi_data/', 'Testing_data', 'testing_dataset_label.pth'))
    #
    # print('haha')
    return dl_tr, dl_ev

@torch.no_grad()
def feature_extract(model, dl):
    device = 'cuda'
    model.eval()
    embeddings = torch.tensor([])
    labels = torch.tensor([])
    for ct, (x, y, _) in tqdm(enumerate(dl)):
        x = x.to(device)
        feat = model(x)
        embeddings = torch.cat((embeddings, feat.detach().cpu()), dim=0)
        labels = torch.cat((labels, y), dim=0)
    return embeddings, labels


if __name__ == '__main__':

    dl_tr, dl_ev = prepare_data()
    feat = bninception()
    in_sz = feat(torch.rand(1, 3, 256, 256)).squeeze().size(0)
    feat.train()
    emb = torch.nn.Linear(in_sz, 512)
    model = torch.nn.Sequential(feat, emb)
    model = nn.DataParallel(model)
    model.cuda()


    for e in range(40):

        model.load_state_dict(torch.load('dvi_data/Model/Epoch_{}/subject_model.pth'.format(e+1)))
        proxies = torch.load('dvi_data/Model/Epoch_{}/proxy.pth'.format(e+1), map_location='cpu').detach().numpy()
        print(proxies.shape)
        # print(model)

        embedding, label = feature_extract(model, dl_tr)
        torch.save(embedding, 'dvi_data/Model/Epoch_{}/training_embeddings.pth'.format(e+1))
        torch.save(label, 'dvi_data/Model/Epoch_{}/training_labels.pth'.format(e+1))

        # Parametric Umap model
        # embedder = ParametricUMAP(encoder=encoder, dims=(512,))
        embedder = ParametricUMAP()

        if e != 0:
            # initialize by last visualization model
            embedder.encoder = tf.keras.models.load_model('dvi_data/Model/Epoch_{}/parametric_model/encoder'.format(e))
        embedder.fit_transform(embedding) # but train on all samples
        embedder.encoder.save('dvi_data/Model/Epoch_{}/parametric_model/encoder'.format(e+1))
        print('haha')

        # only visualize 10 classes
        embedding_sub = embedding[label < 10, :].numpy()
        label_sub = label[label < 10].numpy()
        low_dim_emb = embedder.transform(embedding_sub)
        low_dim_proxy = embedder.transform(proxies)
        # print(low_dim_emb.shape)

        fig = plt.figure()
        plt.scatter(low_dim_emb[:, 0], low_dim_emb[:, 1], c=label_sub, cmap='tab10', s=5)
        plt.scatter(proxies[:10, 0], proxies[:10, 1], c=range(10), cmap='tab10', marker=(5,1))
        fig.savefig('dvi_data/umap_plots/Epoch_{}.png'.format(e+1))
        # break

    # print('haha')
