import torch

import dataset
import utils

import os

import matplotlib
matplotlib.use('agg', force=True)
import random
from tqdm import tqdm
# from apex import amp
from torch.utils.data import Dataset
from loss import *
from utils import predict_batchwise

os.environ["CUDA_VISIBLE_DEVICES"]="0"



if __name__ == '__main__':
    batch_size = 32
    num_cls_per_batch = 8
    sz_embedding = 2048
    config = utils.load_config('config/cub_dist.json')

    # set random seed for all gpus
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # load training dataset CUB200 dataset
    dataset_config = utils.load_config('dataset/config.json')
    train_transform = dataset.utils.make_transform(
                **dataset_config['transform_parameters']
            )
    tr_dataset = dataset.load(
        name='cub',
        root=dataset_config['dataset']['cub']['root'],
        source=dataset_config['dataset']['cub']['source'],
        classes=dataset_config['dataset']['cub']['classes']['trainval'],
        transform=train_transform
    )

    print('Length of dataset: ', len(tr_dataset))

    # dataloader
    # balanced batch sampler, in each batch, randomly sample 8 classes each has 4 images
    batch_sampler = dataset.utils.BalancedBatchSampler(torch.Tensor(tr_dataset.ys), num_cls_per_batch,
                                                       int(batch_size / num_cls_per_batch))
    dl_tr = torch.utils.data.DataLoader(
        tr_dataset,
        batch_sampler = batch_sampler,
        num_workers = 0,
        shuffle=False, # TODO: disable shuffling for debugging purpose
    )

    # training dataloader without shuffling and without transformation
    dl_tr_noshuffle = torch.utils.data.DataLoader(
            dataset=dataset.load(
                    name='cub',
                    root=dataset_config['dataset']['cub']['root'],
                    source=dataset_config['dataset']['cub']['source'],
                    classes=dataset_config['dataset']['cub']['classes']['trainval'],
                    transform=dataset.utils.make_transform(
                        **dataset_config['transform_parameters'],
                        is_train=False
                    )
                ),
            num_workers = 0,
            shuffle=False,
            batch_size=64,
            # sampler=sampler,
    )
    print('Length of dataset: ', len(dl_tr.dataset))
    print('Length of dataset: ', len(dl_tr_noshuffle.dataset))


    # model
    feat = config['model']['type']()
    feat.eval()
    in_sz = feat(torch.rand(1, 3, 256, 256)).squeeze().size(0)
    feat.train()
    emb = torch.nn.Linear(in_sz, sz_embedding)
    model = torch.nn.Sequential(feat, emb)

    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # load loss
    # criterion = config['criterion']['type'](
    #     nb_classes = dl_tr.dataset.nb_classes(),
    #     sz_embed = sz_embedding,
    #     len_training = len(dl_tr.dataset),
    #     initial_proxy_num=1,
    #     **config['criterion']['args']
    # ).cuda()
    criterion = ProxyNCA_distribution_loss(nb_classes = dl_tr.dataset.nb_classes(),
                                 sz_embed=sz_embedding,
                                **config['criterion']['args']).cuda()

    # load optimizer
    opt = config['opt']['type'](
        [
            {
                **{'params': list(feat.parameters()
                    )
                },
                **config['opt']['args']['backbone']
            },
            {
                **{'params': list(emb.parameters()
                    )
                },
                **config['opt']['args']['embedding']
            },

            {
                **{'params': criterion.proxies}
                ,
                **config['opt']['args']['proxynca']

            },
            {
                **{'params': criterion.sigmas_inv},
                **config['opt']['args']['proxynca_sigma']
            },

        ],
        **config['opt']['args']['base']
    )

    X, T, *_ = predict_batchwise(model, dl_tr_noshuffle)
    criterion.kmeans_init(X, T)

    # training!
    losses = []
    visited_indices = []
    for e in range(0, 100): # train for 5 epochs for example
        losses_per_epoch = []
        for ct, (x, y, indices) in tqdm(enumerate(dl_tr)):
            x = x.cuda()
            m = model(x)
    #         # FIXME: loss not improving
    #
            loss = criterion(m, indices, y.cuda())
            print(loss)  #
            opt.zero_grad()
            print(criterion.sigmas_inv.grad)
            loss.backward()
            opt.step()
            print(criterion.sigmas_inv.grad)

            print('haha')

