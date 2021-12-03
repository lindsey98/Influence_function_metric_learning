#! /usr/bin/env python3

import torch
from torch.autograd import grad
from tqdm import tqdm
from torch.autograd import Variable


def jacobian(z, t, model, criterion):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.
    Arguments:
        z: torch tensor, training data points
            e.g. an image sample (batch_size, 3, 256, 256)
        t: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        gpu: int, device id to use for GPU, -1 for CPU
    Returns:
        grad_z: list of torch tensor, containing the gradients from model parameters to loss"""
    model.eval(); model.zero_grad()
    criterion.zero_grad(); criterion.proxies.requires_grad = False
    # initialize
    z, t = z.cuda(), t.cuda()
    m = model(z) # get (sz_embed,) feature embedding
    loss = criterion(m, None, t)
    # Compute sum of gradients from model parameters to loss
    params = [p for p in model.module[-1].parameters() if p.requires_grad ] # last linear layer
    return list(grad(loss, params, create_graph=True))


def grad_alltrain(model, criterion, dl_tr):
    grad_all = []
    for ct, (x, t, _) in tqdm(enumerate(dl_tr)):
        grad_this = jacobian(x, t, model, criterion)
        grad_this = [g.detach().cpu() for g in grad_this]
        grad_all.append(grad_this) # (N_tr, |\theta|)
    return grad_all


def inverse_hessian_product(model, criterion, v, dl_tr, scale=10, damping=0.0):
    """
    s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, stochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.
    Arguments:
        model: torch NN, model used to evaluate the dataset
        criterion: loss function
        v: test gradients
        dl_tr: torch Dataloader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor
    Returns:
        h_estimate: list of torch tensors, s_test"""
    # h_estimate
    cur_estimate = v.copy()
    ct = 0
    for _, (x, t, _) in tqdm(enumerate(dl_tr)): # I change to loop over all training samples
        x, t = x.cuda(), t.cuda()
        model.zero_grad()
        criterion.zero_grad(); criterion.proxies.requires_grad = False
        m = model(x)
        loss = criterion(m, None, t)
        params = [p for p in model.module[-1].parameters() if p.requires_grad]
        hv = hessian_vector_product(loss, params, cur_estimate)
        # Inverse Hessian product Update: v + (I - Hessian_at_x) * cur_estimate
        cur_estimate = [_v + (1 - damping) * _h_e - _hv / scale for _v, _h_e, _hv in zip(v, cur_estimate, hv)]
        ct += 1

    inverse_hvp = [b / scale for b in cur_estimate] # rescale it
    inverse_hvp = [a / ct for a in inverse_hvp] # take average
    return inverse_hvp

def calc_confusion(feat_cls1, feat_cls2, sqrt=False):
    n1, n2 = feat_cls1.size()[0], feat_cls2.size()[0]
    inter_dis = ((feat_cls1.mean(0) - feat_cls2.mean(0)) ** 2).sum()  # inter class distance
    intra_dist1 = ((feat_cls1 - feat_cls1.mean(0)) ** 2).sum() / (n1 - 1)  # unbiased estimate of intra-class variance
    intra_dist2 = ((feat_cls2 - feat_cls2.mean(0)) ** 2).sum() / (n2 - 1)
    confusion = inter_dis / (intra_dist1 / n1 + intra_dist2 / n2)
    if sqrt:
        confusion = confusion.sqrt()
    return confusion

def grad_confusion(model, dl_ev, cls1, cls2):
    '''
    Calculate class confusion or gradient of class confusion to model parameters
    Arguments:
        model: torch NN, model used to evaluate the dataset
        dl_ev: test dataloader
        cls1: class 1
        cls2: class 2
        get_grad: compute gradient or the original
    Returns:
    '''

    feat_cls1 = torch.tensor([]).cuda()  # (N1, sz_embed)
    feat_cls2 = torch.tensor([]).cuda()  # (N2, sz_embed)
    model.zero_grad()
    for ct, (x, t, _) in tqdm(enumerate(dl_ev)): # need to recalculate the feature embeddings since we need the gradient
        if t.item() == cls1: # belong to class 1
            x = x.cuda()
            m = model(x)
            feat_cls1 = torch.cat((feat_cls1, m), dim=0)
        elif t.item() == cls2: # belong to class 2
            x = x.cuda()
            m = model(x)
            feat_cls2 = torch.cat((feat_cls2, m), dim=0)
        else: # skip irrelevant test samples
            pass

    confusion = calc_confusion(feat_cls1, feat_cls2, sqrt=False)
    params = [p for p in model.module[-1].parameters() if p.requires_grad]
    return list(grad(confusion, params))

def hessian_vector_product(y, x, v):
    """Multiply the Hessians of y and x by v.
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
        ValueError: `y` and `v` have a different length."""
    if len(x) != len(v):
        raise(ValueError("w and v must have the same length."))
    # First backprop
    first_grads = grad(y, x, retain_graph=True, create_graph=True)
    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem.detach()) # v is considered as constant
    # Second backprop
    return_grads = grad(elemwise_products, x)

    return return_grads


def calc_influential_func(inverse_hvp, grad_alltrain):
    influence_values = []
    n_train = len(grad_alltrain)
    for grad1train in grad_alltrain:
        influence_thistrain = [torch.dot(x.T, y) * (1/n_train) for x, y in zip(inverse_hvp, grad1train)]
        influence_values.append(influence_thistrain)
    return influence_values