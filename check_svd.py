import matplotlib.pyplot as plt
import numpy as np

import dataset
import evaluation.neighborhood
import loss
import utils
import os
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import collections
import numpy.linalg as LA
os.environ["CUDA_VISIBLE_DEVICES"]="1,0"

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)

def _find_z(model, criterion, inputs, targets, h):
    '''
    Finding the direction in the regularizer
    '''
    inputs.requires_grad_()
    outputs = model.eval()(inputs)
    loss_z = criterion(outputs, None, targets)
    loss_z.backward() # backward the gradient

    grad = inputs.grad.data + 0.0
    norm_grad = grad.norm().item() # gradient norm

    z = torch.sign(grad).detach() + 0. # sign of gradient
    z = 1. * h * (z + 1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None] + 1e-7) # sign of gradient / norm(sign of gradient)
    zero_gradients(inputs)
    model.zero_grad()
    criterion.zero_grad()
    return z, norm_grad


def curvature(model, criterion, inputs, targets, h=3.):

    z, norm_grad = _find_z(model, criterion, inputs, targets, h)
    torch.cuda.empty_cache()

    inputs.requires_grad_()
    outputs_pos = model.eval()(inputs + z) # f(x+hz)
    outputs_orig = model.eval()(inputs) # f(x)

    loss_pos = criterion(outputs_pos, None, targets) # Loss(f(x+hz))
    loss_orig = criterion(outputs_orig, None, targets) # Loss(f(x))
    grad_diff = torch.autograd.grad((loss_pos - loss_orig), inputs,
                                    create_graph=False)[0] # grad[Loss(f(x+hz))] - grad[Loss(f(x))]
    reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1) # ||grad[Loss(f(x+hz))] - grad[Loss(f(x))]||_2^2
    model.zero_grad()
    criterion.zero_grad()

    return reg, norm_grad

# def _get_weights(model):
#     W = [model[k].detach().cpu() for k in model.keys() if ('fc' in k) or ('conv' in k)]
#     weights = [w.view(w.size()[0], -1) for w in W]
#     return weights

def normalized_margin(model, criterion, loader):
    # S = prod(max(sigma(A)) * L_i)
    # T = sum_i ||A_i - M_i]^{2/3}_1 / || ||A_i||^{2/3}_2
    # R = T**(3/2) * S
    # Lipschitz constants: relu: 1. max_pooling lipschitz: 1. log-softmax: 1
    # TODO: calculate margin with *all* data points
    # A = _get_weights(torch.load(model_weigths_path))
    # L2norms = [LA.norm(a.data.numpy(), ord=2) for a in A]
    # L1norms = [LA.norm(a.data.numpy().flat[:], ord=1) for a in A]
    #
    # T = sum(l1 ** (2 / 3) / l2 ** (2 / 3) for l1, l2 in zip(L1norms, L2norms))
    # S = np.prod(L2norms)
    # R = T ** (3 / 2) * S
    # n = len(loader.dataset)

    margins = torch.Tensor()
    for data, target, *_ in tqdm(loader):
        data, target = data.cuda(), target.cuda()
        output = model(data)
        _, margin = criterion.debug(output, None, target)
        margin = margin.detach().cpu()
        margins = torch.cat((margins, margin))

    margin_dist = margins

    return margin_dist

if __name__ == '__main__':
    sz_embedding = 512
    dataset_name = 'cars'
    loss_type = 'ProxyNCA_pfix_test'

    for seed in range(0, 1):
        print(seed)
        model_dir = 'dvi_data_{}_{}_loss{}/ResNet_512_Model/Epoch_40/{}_{}_trainval_512_{}.pth'.format(dataset_name, seed, loss_type,
                                                                                                       dataset_name, dataset_name, seed)
        proxy_dir = 'dvi_data_{}_{}_loss{}/ResNet_512_Model/Epoch_40/proxy.pth'.format(dataset_name, seed, loss_type)
        config = utils.load_config('config/cars.json')

        # load training dataset
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
                batch_size=32,
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

        # criterion
        criterion = loss.ProxyNCA_pfix(nb_classes=dl_tr_noshuffle.dataset.nb_classes(),
                                       sz_embed=sz_embedding,
                                       scale=3)
        criterion.proxies.data = torch.load(proxy_dir)['proxies']
        # evaluation.neighborhood.neighboring_emb_finding(model, dl_tr=dl_tr_noshuffle, dl_ev=dl_ev)
        # calculate embeddings with model and get targets
        X_train, T_train, *_ = utils.predict_batchwise(model, dl_tr_noshuffle)
        # X_test, T_test, *_ = utils.predict_batchwise(model, dl_ev)
        # print('done collecting prediction')
        # for length_scale in np.linspace(0.5, 1, 10):
        #     for weight in np.linspace(0.1, 0.5, 10):
        #         print(length_scale, weight)
        #         evaluation.neighborhood.evaluate_neighborhood(model, X_train, X_test, T_test, weight=weight, length_scale=length_scale)
        # print('Training')
        # print(utils.evaluate(model, dl_tr_noshuffle))
        # print('Testing')
        # print(utils.evaluate(model, dl_ev))
        #
        # proxies = torch.load(proxy_dir)['proxies']
        # avg_inter_proxy_ip, var_inter_proxy_ip = utils.inter_proxy_dist(proxies, cosine=True, neighbor_only=False)
        # print('Proxy-Proxy')
        # print(avg_inter_proxy_ip.item())
        # print(var_inter_proxy_ip.item())
        #
        # avg_inter_proxy_ip, var_inter_proxy_ip = utils.inter_proxy_dist(proxies, cosine=True, neighbor_only=True)
        # print('Proxy-Neighbor')
        # print(avg_inter_proxy_ip.item())
        # print(var_inter_proxy_ip.item())

        '''Get curvature'''
        # curvatures = torch.tensor([])
        # for ct, (inputs, targets, indices) in tqdm(enumerate(dl_tr_noshuffle)):
        #     inputs = inputs.cuda()
        #     targets = targets.cuda()
        #     cur_reg, _ = curvature(model, criterion, inputs, targets, h=3.)
        #     curvatures = torch.cat((curvatures, cur_reg.detach().cpu()))
        #
        # print('Average curvature around training points: {}'.format(torch.mean(curvatures).item()))

        '''Get normalized margin'''
        # margin_dist = normalized_margin(model, criterion, dl_tr_noshuffle)
        # print('Average top1-top2 margin:', margin_dist.mean().item())
        #
        '''Get rho'''
        # rho = utils.get_rho(X_train)
        # print("Rho (smaller rho has more directions of significant variance):", rho)
        #
        '''Get feature dependence'''
        # feat_corr = torch.corrcoef(X_train.detach().cpu().T)
        # non_diag = torch.ones_like(feat_corr).to(feat_corr.device) - torch.eye(len(feat_corr)).to(feat_corr.device)
        # reduced_corr_mat = feat_corr * non_diag # mask diagonal
        # print('Frobenius norm', torch.norm(reduced_corr_mat).item())
        # print('Average between feature correlation', reduced_corr_mat.mean().item())
        # triu_indices = torch.triu_indices(len(feat_corr), len(feat_corr), 1)
        # inter_feat_corr = reduced_corr_mat[triu_indices[0, :], triu_indices[1, :]]
        # print('Maximum inter-feature correlation:', inter_feat_corr.flatten().max())
        # print('Minimum inter-feature correlation:', inter_feat_corr.flatten().min())
        #
        # plt.hist(inter_feat_corr.flatten().numpy().tolist(), bins=50)
        # plt.show()

        '''Get RFF feature dependence'''
        rff_feat_cov = utils.get_RFF_cov(X_train)
        print('Frobenius norm on RFF feature cov', rff_feat_cov)

        '''Embedding space density (Avg Intra/ Avg Inter)'''
        # intra_inter_ratio, reduced_dist_mat = utils.get_intra_inter_dist(X_train, T_train, dl_tr_noshuffle.dataset.nb_classes())
        # print('Average intra/inter ratio:', intra_inter_ratio)
        # plt.hist(reduced_dist_mat.tolist(), bins=50)
        # plt.title('Distribution of inter-class distance')
        # plt.show()

