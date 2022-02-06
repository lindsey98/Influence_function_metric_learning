#! /usr/bin/env python3
# Code is modified from https://github.com/kohpangwei/influence-release and https://github.com/nimarb/pytorch_influence_functions
import pickle

import torch
from torch.autograd import grad
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np

# def grad_loss_relabel(model, criterion, all_features, all_labels, dl):
#     '''
#         Get dL/dtheta for all training
#     '''
#     grad_all = []
#     model.eval(); model.zero_grad()  # first zero out previous gradients
#     criterion.zero_grad(); criterion.proxies.requires_grad = False  # first zero out previous gradients, set proxies to be non-differentiable
#
#     for feat, t in zip(all_features, all_labels):
#         x = feat.cuda().unsqueeze(0)
#         x = torch.repeat_interleave(x, repeats=dl.dataset.nb_classes(), dim=0)
#         gradient_this = torch.tensor([])
#         # for y in torch.arange(dl.dataset.nb_classes()):
#         model.zero_grad()
#         m = model.module[-1](x)
#         # loss = criterion(m, None, y.view(1, -1))
#         y = torch.arange(dl.dataset.nb_classes()).cuda()
#         loss = criterion.debug(m, None, y)
#         params = model.module[-1].weight  # last linear layer weights
#         gradient = list(grad(loss, params, create_graph=False))[0].detach().cpu() # (512, 2048)
#         gradient_this = torch.cat([gradient_this, gradient.unsqueeze(0)], dim=0)  # (C, 512, 2048)
#         gradient_this = gradient_this - gradient_this[t.long().item()] # (C, 512, 2048)
#         grad_all.append(gradient_this) # (N_tr, C, 512, 2048)
#     return grad_all
# def calc_influential_func_relabel(IS, train_features, inverse_hvp_prod):
#     '''
#         Calculate influential functions
#         Arguments:
#             inverse_hvp_prod: inverse hessian vector product, of shape (|theta|,)
#             grad_alltrain: list of gradients for all training (N_train, |theta|)
#         Returns:
#             influence_values: list of influence values (N_train,)
#     '''
#     influence_values = []
#     grad4train = grad_loss_relabel(IS.model, IS.criterion, train_features, IS.train_label, IS.dl_tr)
#     for i in tqdm(range(len(train_features))):
#         # influence = (-1) * grad(test)' H^-1 grad(train), dont forget the negative sign
#         grad1train = grad4train[i]
#         influence_thistrain = [(-1) * torch.dot(x.flatten().detach().cpu(), y.flatten()).item() \
#                                for x, y in zip(inverse_hvp_prod * grad1train.size()[0], grad1train)] # (C, )
#         influence_values.append(influence_thistrain) # (N_tr, C)
#     return influence_values

def grad_loss(model, criterion, all_features, all_labels):
    '''
        Get dL/dtheta for all training
        Arguments:
            model: model
            criterion: loss
            all_features: features (N, 2048)
            all_labels: labels (N,)
        Returns:
            List of dLoss/dtheta
    '''
    grad_all = []
    for feat, t in zip(all_features, all_labels):
        feat, t = feat.unsqueeze(0).cuda(), t.view(1, -1).cuda()
        model.eval(); model.zero_grad()  # first zero out previous gradients
        criterion.zero_grad(); criterion.proxies.requires_grad = False  # first zero out previous gradients, set proxies to be non-differentiable

        m = model.module[-1](feat)  # get (sz_embed,) feature embedding
        loss = criterion(m, None, t)

        params = model.module[-1].weight  # last linear layer weights
        grad_this = list(grad(loss, params, create_graph=True)) # gradient
        grad_this = [g.detach().cpu() for g in grad_this]
        grad_all.append(grad_this) # (N_tr, |\theta|)
    return grad_all


def inverse_hessian_product(model, criterion, v, dl_tr,
                            scale=500, damping=0.01):
    """
        Get grad(test)' H^-1 grad(train). v is grad(test)
        Arguments:
            model: torch NN, model used to evaluate the dataset
            criterion: loss function
            v: vector you want to multiply with H-1
            dl_tr: torch Dataloader, can load the training dataset
            damping: float, dampening factor "chosen to be roughly the size of the most negative eigenvalue of the empirical Hessian (so that it becomes PSD)."
            scale: float, scaling factor, "the scale parameter scales the maximum eigenvalue to < 1 so that our Taylor approximation converges, otherwise h_estimate get NaN"
        Returns:
            h_estimate: list of torch tensors, s_test
    """
    cur_estimate = v.copy() #
    for ct, (x, t, _) in tqdm(enumerate(dl_tr)): # I change it to be looping over all training samples
        x, t = x.cuda(), t.cuda()
        model.eval(); model.zero_grad()
        criterion.zero_grad(); criterion.proxies.requires_grad = False
        m = model(x)
        loss = criterion(m, None, t)
        params = [model.module[-1].weight]
        hv = hessian_vector_product(loss, params, cur_estimate) # get hvp
        # Inverse Hessian product Update: v + (I - Hessian_at_x) * cur_estimate
        cur_estimate = [_v + (1 - damping) * _h_e - _hv.detach().cpu() / scale for _v, _h_e, _hv in zip(v, cur_estimate, hv)]
        pass

    inverse_hvp = [b.detach().cpu() / scale for b in cur_estimate] # "In the loop, we scale the Hessian down by scale, which means that the estimate of the inverse Hessian-vector product will be scaled up by scale. The last division corrects for this scaling."
    return inverse_hvp # I didn't divide it by number of recursions

def hessian_vector_product(y, x, v):
    """
        Multiply the Hessians of y and x by v.
        Uses a backprop-like approach to compute the product between the Hessian
        and another vector efficiently, which even works for large Hessians.
        Example: if: y = 1/2 x'Ax then hvp(y, x, v) returns and expression
        which evaluates to the same values as (A + A.t) v.
        Arguments:
            y: scalar/tensor, for example the output of the loss function
            x: list of torch tensors, tensors over which the Hessian
                should be constructed
            v: list of torch tensors, same shape as w,
                will be multiplied with the Hessian
        Returns:
            return_grads: list of torch tensors, contains product of Hessian and v.
        Raises:
            ValueError: `y` and `v` have a different length.
    """
    if len(x) != len(v):
        raise(ValueError("w and v must have the same length."))
    # First backprop
    first_grads = list(grad(y, x, retain_graph=True, create_graph=True))
    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem.to(grad_elem.device).detach()) # v is considered as constant
    # Second backprop
    return_grads = grad(elemwise_products, x)
    return return_grads

def calc_influential_func_orig(IS, train_features, inverse_hvp_prod):
    '''
        Calculate influential functions
        Arguments:
            inverse_hvp_prod: inverse hessian vector product, of shape (|theta|,)
            train_features: training features (N_trauin, 2048)
        Returns:
            influence_values: list of influence values (N_train,)
    '''
    influence_values = []
    for i in tqdm(range(len(train_features))):
        # influence = (-1) * grad(test)' H^-1 grad(train), dont forget the negative sign
        grad1train = grad_loss(IS.model, IS.criterion, [train_features[i]], [IS.train_label[i]])[0]
        influence_thistrain = [(-1) * torch.dot(x.flatten().detach().cpu(), y.flatten()).item() \
                               for x, y in zip(inverse_hvp_prod, grad1train)]
        influence_values.append(influence_thistrain)
    return influence_values




