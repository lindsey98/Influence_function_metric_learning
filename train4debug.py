import dataset
import utils

import os

import matplotlib
from tqdm import tqdm
from torch.utils.data import Dataset
import loss
from networks import *
import torch
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="0"


if __name__ == '__main__':
    batch_size = 16
    num_cls_per_batch = 8
    sz_embedding = 512
    config = utils.load_config('config/cub.json')

    # set random seed for all gpus
    # random.seed(0)
    # np.random.seed(0)
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

    # os.makedirs('/home/ruofan/PycharmProjects/SoftTriple/datasets/vgg_face_dataset/train', exist_ok=True)
    # for path in tr_dataset.im_paths:
    #     class_name = os.path.dirname(path).split('/')[-1]
    #     try:
    #         rgba_image = PIL.Image.open(path)
    #         rgb_image = rgba_image.convert('RGB')
    #         rgb_image.save(path)
    #     except PIL.UnidentifiedImageError:
    #         os.unlink(path)
    #         os.unlink(os.path.join('/home/ruofan/PycharmProjects/SoftTriple/datasets/vgg_face_dataset/train',
    #                                class_name,
    #                                os.path.basename(path)))
    #         print(path)
        # os.makedirs(os.path.join('/home/ruofan/PycharmProjects/SoftTriple/datasets/vgg_face_dataset/train', class_name), exist_ok=True)
        # shutil.copyfile(path, os.path.join('/home/ruofan/PycharmProjects/SoftTriple/datasets/vgg_face_dataset/train',
        #                                    class_name,
        #                                    os.path.basename(path)))
    ev_dataset =  dataset.load(
            name='cub',
            root=dataset_config['dataset']['cub']['root'],
            source=dataset_config['dataset']['cub']['source'],
            classes=dataset_config['dataset']['cub']['classes']['eval'],
            transform=dataset.utils.make_transform(
                **dataset_config['transform_parameters'],
                is_train=False
            )
    )

    # os.makedirs('/home/ruofan/PycharmProjects/SoftTriple/datasets/vgg_face_dataset/test', exist_ok=True)
    # for path in ev_dataset.im_paths:
    #     class_name = os.path.dirname(path).split('/')[-1]
    #     try:
    #         rgba_image = PIL.Image.open(path)
    #         rgb_image = rgba_image.convert('RGB')
    #         rgb_image.save(path)
    #     except PIL.UnidentifiedImageError:
    #         os.unlink(path)
    #         os.unlink(os.path.join('/home/ruofan/PycharmProjects/SoftTriple/datasets/vgg_face_dataset/test',
    #                                class_name,
    #                                os.path.basename(path)))
    #         print(path)
        # os.makedirs(os.path.join('/home/ruofan/PycharmProjects/SoftTriple/datasets/vgg_face_dataset/test', class_name), exist_ok=True)
        # shutil.copyfile(path, os.path.join('/home/ruofan/PycharmProjects/SoftTriple/datasets/vgg_face_dataset/test',
        #                                    class_name,
        #                                    os.path.basename(path)))
    # print('Length of training dataset: ', len(tr_dataset))
    # print('Length of testing dataset: ', len(dl_ev.dataset))

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
    #
    # # training dataloader without shuffling and without transformation
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
    #
    #
    # # model
    # feat = config['model']['type']()
    # feat.eval()
    # in_sz = feat(torch.rand(1, 3, 256, 256))[0].squeeze().size(0)
    # feat.train()
    # emb = torch.nn.Linear(in_sz, sz_embedding)
    # model = torch.nn.Sequential(feat, emb)

    from loss_with_model import Encoder
    model = Encoder(100)
    # model = torch.nn.DataParallel(model)
    model = model.cuda()
    print(list(model.parameters()))
    #
    # # load loss
    # criterion = config['criterion']['type'](
    #     nb_classes = dl_tr.dataset.nb_classes(),
    #     sz_embed = sz_embedding,
    #     len_training = len(dl_tr.dataset),
    #     initial_proxy_num=1,
    #     **config['criterion']['args']
    # ).cuda()
    # criterion = loss.ProxyNCA_prob_dynamic(nb_classes = dl_tr.dataset.nb_classes(),
    #                              sz_embed=sz_embedding,
    #                             **config['criterion']['args']).cuda()

    # load optimizer
    # opt = config['opt']['type'](
    #     [
    #         {
    #             **{'params': list(feat.parameters()
    #                 )
    #             },
    #             **config['opt']['args']['backbone']
    #         },
    #         {
    #             **{'params': list(emb.parameters()
    #                 )
    #             },
    #             **config['opt']['args']['embedding']
    #         },
    #
    #         {
    #             **{'params': criterion.proxies}
    #             ,
    #             **config['opt']['args']['proxynca']
    #
    #         },
    #         # {
    #         #     **{'params': criterion.sigmas_inv},
    #         #     **config['opt']['args']['proxynca_sigma']
    #         # },
    #
    #     ],
    #      **config['opt']['args']['base']
    # )

    # load optimizer
    opt = config['opt']['type'](
        [
            {
                **{'params': list(model.base.parameters()) + list(model.emb.parameters()) + list(model.lnorm.parameters())
                },
                **config['opt']['args']['backbone']
            },

            {
                **{'params': model.proxies}
                ,
                **config['opt']['args']['proxynca']

            },

        ],
         **config['opt']['args']['base']
    )

    # model.load_state_dict(torch.load('results/cub_cub_trainval_512_0_False_lossProxyNCA_prob_orig_proxy1_tau0.00.pt'))
    # criterion.proxies.data = torch.load('checkpoints/dvi_data_cub_False_lossProxyNCA_prob_orig_proxy1_tau0.0/ResNet_512_Model/Epoch_31/proxy.pth')['proxies']

    # training!
    losses = []
    visited_indices = []
    for e in range(0, 100): # train for 5 epochs for example
        losses_per_epoch = []
        for ct, (x, y, indices) in tqdm(enumerate(dl_tr)):
            x = x.cuda()
            y = y.cuda()
            loss = model.calc_loss(x, y)
            print(loss)  #

            # FIXME: loss not improving

            # loss = criterion(m, indices, y.cuda())
            # smoothness regularizer
            # print(loss)  #
            opt.zero_grad()
            # print(criterion.sigmas_inv.grad)
            loss.backward()
            opt.step()
            # print(criterion.sigmas_inv.grad)
            #
            # print('haha')

