import matplotlib.pyplot as plt

import dataset
import utils

import os

from tqdm import tqdm
from torch.utils.data import Dataset
import torch

os.environ["CUDA_VISIBLE_DEVICES"]="0"


if __name__ == '__main__':
    sz_embedding = 512
    dataset_name = 'cars'
    model_dir = 'results/cars_cars_mixup_trainval_512_0_lossProxyNCA_prob_mixup_both_weighted_with_classsamplerdiverse.pt'
    proxy_dir = 'dvi_data_inshop_lossProxyNCA_prob_mixup_interproxy_random/ResNet_512_Model/Epoch_40/proxy.pth'
    config = utils.load_config('config/cars.json')

    # set random seed for all gpus
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

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

    # batch_sampler = dataset.utils.BalancedBatchSampler(torch.Tensor(tr_dataset.ys), num_cls_per_batch,
    #                                                    int(batch_size / num_cls_per_batch))
    # dl_tr = torch.utils.data.DataLoader(
    #     tr_dataset,
    #     batch_sampler = batch_sampler,
    #     num_workers = 0,
    #     shuffle=False, # TODO: disable shuffling for debugging purpose
    # )
    # #
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
    # plt.plot(svd.detach().cpu().numpy())
    # for i, j in zip(range(len(svd)), svd):
    #     plt.annotate(str(round(j.item(), 2)), xy=(i, j))
    # plt.title(model_dir.split('/')[-1].split('.pt')[0])
    # plt.show()

    proxies = torch.load(proxy_dir)['proxies']
    avg_inter_proxy_ip = utils.inter_proxy_dist(proxies)
    print(avg_inter_proxy_ip.item())

