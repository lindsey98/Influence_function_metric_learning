#! /usr/bin/env python3

import torch
from torch.autograd import grad
from tqdm import tqdm
from torch.autograd import Variable

def grad_alltrain(model, criterion, dl_tr):
    grad_all = []
    for ct, (x, t, _) in tqdm(enumerate(dl_tr)):
        grad_this = grad_z(x, t, model, criterion)
        grad_this = [g.detach().cpu() for g in grad_this]
        grad_all.append(grad_this) # (N_tr, |\theta|)
    return grad_all

def s_test(model, criterion, v, dl_tr, damp=0.01, scale=25.0):
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
    torch.cuda.empty_cache()
    h_estimate = v.copy()
    for ct, (x, t, _) in tqdm(enumerate(dl_tr)): # I change to loop over all training samples
        x, t = x.cuda(), t.cuda()
        model.zero_grad()
        criterion.zero_grad(); criterion.proxies.requires_grad = False
        m = model(x)
        loss = criterion(m, None, t)
        params = [p for p in model.module[-1].parameters() if p.requires_grad]
        hv = hessian_vector_product(loss, params, h_estimate)
        # Recursively caclulate h_estimate
        h_estimate = [_v + (1 - damp) * _h_e - _hv / scale for _v, _h_e, _hv in zip(v, h_estimate, hv)]

    return h_estimate

def grad_confusion(model, dl_ev, cls1, cls2, get_grad=False):
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
    feat_cls1 = torch.tensor([]).cuda() # (N1, sz_embed)
    feat_cls2 = torch.tensor([]).cuda()  # (N2, sz_embed)
    model.zero_grad()
    for ct, (x, t, _) in tqdm(enumerate(dl_ev)):
        if t.item() == cls1:
            x = x.cuda()
            m = model(x)
            feat_cls1 = torch.cat((feat_cls1, m), dim=0)
            pass
        elif t.item() == cls2:
            x = x.cuda()
            m = model(x)
            feat_cls2 = torch.cat((feat_cls2, m), dim=0)
            pass
        else:
            pass

    inter_dis = ((feat_cls1.mean(0) - feat_cls2.mean(0))**2).sum() # inter class distance
    intra_dist1 = ((feat_cls1 - feat_cls1.mean(0))**2).sum() / (feat_cls1.size()[0] - 1) # unbiased estimate of intra-class variance
    intra_dist2 = ((feat_cls2 - feat_cls2.mean(0))**2).sum() / (feat_cls2.size()[0] - 1)
    confusion = inter_dis / (intra_dist1/feat_cls1.size()[0] + intra_dist2/feat_cls2.size()[0])

    if get_grad:
        params = [p for p in model.module[-1].parameters() if p.requires_grad]
        return list(grad(confusion, params, create_graph=True))
    else:
        return confusion

def grad_z(z, t, model, criterion):
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
    return_grads = grad(elemwise_products, x, create_graph=True)

    return return_grads