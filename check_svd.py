import matplotlib.pyplot as plt

import dataset
import utils

import os

from tqdm import tqdm
from torch.utils.data import Dataset
import torch

os.environ["CUDA_VISIBLE_DEVICES"]="1"


if __name__ == '__main__':
    sz_embedding = 512
    dataset_name = 'cars'
    loss_type = 'ProxyAnchor_pfix_controlvariance'

    for seed in range(6):
        print(seed)
        model_dir = 'dvi_data_{}_{}_loss{}/ResNet_512_Model/Epoch_40/{}_{}_trainval_512_{}.pth'.format(dataset_name, seed, loss_type,
                                                                                                       dataset_name, dataset_name, seed)
        proxy_dir = 'dvi_data_{}_{}_loss{}/ResNet_512_Model/Epoch_40/proxy.pth'.format(dataset_name, seed, loss_type)
        config = utils.load_config('config/cars.json')

        # load training dataset CUB200 dataset
        dataset_config = utils.load_config('dataset/config.json')
        train_transform = dataset.utils.make_transform(
                    **dataset_config['transform_parameters']
        )
        tr_dataset = dataset.load(
            name=dataset_name,
            root=dataset_config['dataset'][dataset_name]['root'],
            source=dataset_config['dataset'][dataset_name]['source'],
            classes=dataset_config['dataset'][dataset_name]['classes']['trainval'],
            transform=train_transform
        )

        # # training dataloader without shuffling and without transformation
        dl_tr_noshuffle = torch.utils.data.DataLoader(
                dataset=dataset.load(
                        name=dataset_name,
                        root=dataset_config['dataset'][dataset_name]['root'],
                        source=dataset_config['dataset'][dataset_name]['source'],
                        classes=dataset_config['dataset'][dataset_name]['classes']['trainval'],
                        transform=dataset.utils.make_transform(
                            **dataset_config['transform_parameters'],
                            is_train=False
                        )
                    ),
                num_workers = 0,
                shuffle=False,
                batch_size=64,
        )
        print('Length of training dataset: ', len(dl_tr_noshuffle.dataset))

        # # training dataloader without shuffling and without transformation
        dl_ev = torch.utils.data.DataLoader(
                dataset=dataset.load(
                        name=dataset_name,
                        root=dataset_config['dataset'][dataset_name]['root'],
                        source=dataset_config['dataset'][dataset_name]['source'],
                        classes=dataset_config['dataset'][dataset_name]['classes']['eval'],
                        transform=dataset.utils.make_transform(
                            **dataset_config['transform_parameters'],
                            is_train=False
                        )
                    ),
                num_workers = 0,
                shuffle=False,
                batch_size=128,
        )
        # model
        feat = config['model']['type']()
        feat.eval()
        in_sz = feat(torch.rand(1, 3, 256, 256))[0].squeeze().size(0)
        feat.train()
        emb = torch.nn.Linear(in_sz, sz_embedding)
        model = torch.nn.Sequential(feat, emb)
        model = torch.nn.DataParallel(model)
        model.cuda()

        # load state_dict
        model.load_state_dict(torch.load(model_dir))

        print('Training')
        print(utils.evaluate(model, dl_tr_noshuffle))
        print('Testing')
        print(utils.evaluate(model, dl_ev))

        proxies = torch.load(proxy_dir)['proxies']
        avg_inter_proxy_ip, var_inter_proxy_ip = utils.inter_proxy_dist(proxies, cosine=True, neighbor_only=False)
        print('Proxy-Proxy')
        print(avg_inter_proxy_ip.item())
        print(var_inter_proxy_ip.item())

        avg_inter_proxy_ip, var_inter_proxy_ip = utils.inter_proxy_dist(proxies, cosine=True, neighbor_only=True)
        print('Proxy-Neighbor')
        print(avg_inter_proxy_ip.item())
        print(var_inter_proxy_ip.item())

