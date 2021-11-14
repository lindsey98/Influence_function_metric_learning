import matplotlib.pyplot as plt

import dataset
import utils

import os

from tqdm import tqdm
from torch.utils.data import Dataset
import torch

os.environ["CUDA_VISIBLE_DEVICES"]="1, 0"


if __name__ == '__main__':
    sz_embedding = 512
    dataset_name = 'cub'

    for seed in range(6):
        print(seed)
        model_dir = 'dvi_data_cub_{}_lossProxyAnchor_pfix/ResNet_512_Model/Epoch_40/cub_cub_trainval_512_{}.pth'.format(seed, seed)
        proxy_dir = 'dvi_data_cub_{}_lossProxyAnchor_pfix/ResNet_512_Model/Epoch_40/proxy.pth'.format(seed)
        config = utils.load_config('config/cub.json')

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

        # get svd
        # svd = utils.get_svd(model, dl_tr_noshuffle, return_avg=True)
        # print(svd.item())
        # plt.plot(svd.detach().cpu().numpy())
        # for i, j in zip(range(len(svd)), svd):
        #     plt.annotate(str(round(j.item(), 2)), xy=(i, j))
        # plt.title(model_dir.split('/')[-1].split('.pt')[0])
        # plt.show()

        # print(utils.evaluate(model, dl_ev))
        proxies = torch.load(proxy_dir)['proxies']
        avg_inter_proxy_ip, var_inter_proxy_ip = utils.inter_proxy_dist(proxies, cosine=True, neighbor_only=True)
        print(avg_inter_proxy_ip.item())
        print(var_inter_proxy_ip.item())

        # gaps = utils.calc_gap(model, dl_tr_noshuffle, proxies)
        # print(gaps)
