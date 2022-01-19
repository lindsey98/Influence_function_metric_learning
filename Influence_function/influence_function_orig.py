#! /usr/bin/env python3
# Code is modified from https://github.com/kohpangwei/influence-release and https://github.com/nimarb/pytorch_influence_functions
import pickle

import torch
from torch.autograd import grad
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np


def grad_loss(model, criterion, all_features, all_labels):
    '''
        Get dL/dtheta for all training
    '''
    grad_all = []
    for feat, t in tqdm(zip(all_features, all_labels)):
        grad_this = jacobian(feat.unsqueeze(0), t.view(1, -1), model, criterion)
        grad_this = [g.detach().cpu() for g in grad_this]
        grad_all.append(grad_this) # (N_tr, |\theta|)
    return grad_all

def jacobian(feat, t, model, criterion):
    """
        Calculates the gradient z. One grad_z should be computed for each training sample.
        Arguments:
            z: torch tensor, training data points
                e.g. an image sample (batch_size, 3, 256, 256)
            t: torch tensor, training data labels
            model: torch NN, model used to evaluate the dataset
            criterion: loss
        Returns:
            grad_z: list of torch tensor, containing the gradients from model parameters to loss
    """
    model.eval(); model.zero_grad() # first zero out previous gradients
    criterion.zero_grad(); criterion.proxies.requires_grad = False # first zero out previous gradients, set proxies to be non-differentiable
    # initialize
    feat, t = feat.cuda(), t.cuda()
    m = model.module[-1](feat) # get (sz_embed,) feature embedding
    loss = criterion(m, None, t)
    # Compute sum of gradients from model parameters to loss
    params = model.module[-1].weight # last linear layer weights
    return list(grad(loss, params, create_graph=True))

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

def calc_influential_func(inverse_hvp_prod, grad_alltrain):
    '''
        Calculate influential functions
        Arguments:
            inverse_hvp_prod: inverse hessian vector product, of shape (|theta|,)
            grad_alltrain: list of gradients for all training (N_train, |theta|)
        Returns:
            influence_values: list of influence values (N_train,)
    '''
    influence_values = []
    for grad1train in grad_alltrain:
        # influence = (-1) * grad(test)' H^-1 grad(train), dont forget the negative sign
        influence_thistrain = [(-1) * torch.dot(x.flatten().detach().cpu(), y.flatten()).item() \
                               for x, y in zip(inverse_hvp_prod, grad1train)]
        influence_values.append(influence_thistrain)
    return influence_values




