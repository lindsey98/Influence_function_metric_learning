#! /usr/bin/env python3
# Code is modified from https://github.com/kohpangwei/influence-release and https://github.com/nimarb/pytorch_influence_functions
import torch
from torch.autograd import grad
from tqdm import tqdm
from torch.autograd import Variable

def jacobian(z, t, model, criterion):
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
    z, t = z.cuda(), t.cuda()
    m = model(z) # get (sz_embed,) feature embedding
    loss = criterion(m, None, t)
    # Compute sum of gradients from model parameters to loss
    params = model.module[-1].weight # last linear layer weights
    return list(grad(loss, params, create_graph=True))

def grad_alltrain(model, criterion, dl_tr, start=None, batch=None):
    '''
        Get gradient for all training set
        Arguments:
            model:
            criterion: loss
            dl_tr: train dataloader
        Returns:
            grad_all: list of shape (N_tr, |\theta|)
    '''
    grad_all = []
    for ct, (x, t, _) in tqdm(enumerate(dl_tr)):
        if start is not None:
            if ct < start: continue
        grad_this = jacobian(x, t, model, criterion)
        grad_this = [g.detach().cpu() for g in grad_this]
        grad_all.append(grad_this)
        if batch is not None:
            if ct - start >= batch - 1: break # support processing a subset of training only
    return grad_all

def calc_intravar(feat):
    '''
        Get intra-class variance (unbiased estimate)
    '''
    n = feat.size()[0]
    intra_var = ((feat - feat.mean(0)) ** 2).sum() / (n - 1)
    return intra_var

def calc_confusion(feat_cls1, feat_cls2, sqrt=False):
    '''
        Calculate class confusion
        Arguments:
            feat_cls1
            feat_cls2
            sqrt
        Returns:
            confusion
    '''
    n1, n2 = feat_cls1.size()[0], feat_cls2.size()[0]
    inter_dis = ((feat_cls1.mean(0) - feat_cls2.mean(0)) ** 2).sum()  # inter class distance
    intra_dist1 = calc_intravar(feat_cls1)  # unbiased estimate of intra-class variance
    intra_dist2 = calc_intravar(feat_cls2)
    confusion = inter_dis / (intra_dist1 / n1 + intra_dist2 / n2) # t^2 statistic
    if sqrt:
        confusion = confusion.sqrt() # t
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
            grad_confusion2params: gradient of confusion to params
    '''
    feat_cls1 = torch.tensor([]).cuda()  # (N1, sz_embed)
    feat_cls2 = torch.tensor([]).cuda()  # (N2, sz_embed)
    model.eval(); model.zero_grad()
    for ct, (x, t, _) in tqdm(enumerate(dl_ev)): # need to recalculate the feature embeddings since we need the gradient
        if t.item() == cls1: # belong to class 1
            if len(feat_cls1) >= 50: # FIXME: feeding in all instances will be out of memory
                continue
            x = x.cuda()
            m = model(x)
            feat_cls1 = torch.cat((feat_cls1, m), dim=0)
        elif t.item() == cls2: # belong to class 2
            if len(feat_cls2) >= 50:
                continue
            x = x.cuda()
            m = model(x)
            feat_cls2 = torch.cat((feat_cls2, m), dim=0)
        if len(feat_cls1) >= 50 and len(feat_cls2) >= 50:
            break
        else: # skip irrelevant test samples
            pass

    confusion = calc_confusion(feat_cls1, feat_cls2, sqrt=False) # d(t^2)/d(theta)
    params = model.module[-1].weight # last linear layer
    grad_confusion2params = list(grad(confusion, params))
    return grad_confusion2params

def grad_intravar(model, dl_ev, cls):
    '''
        Calculate class confusion or gradient of class confusion to model parameters
        Arguments:
            model: torch NN, model used to evaluate the dataset
            dl_ev: test dataloader
            cls1: class 1
            cls2: class 2
            get_grad: compute gradient or the original
        Returns:
            grad_confusion2params: gradient of confusion to params
    '''
    feat_cls = torch.tensor([]).cuda()  # (N, sz_embed)
    model.eval(); model.zero_grad()
    for ct, (x, t, _) in tqdm(enumerate(dl_ev)): # need to recalculate the feature embeddings since we need the gradient
        if t.item() == cls: # belong to class
            x = x.cuda()
            m = model(x)
            feat_cls = torch.cat((feat_cls, m), dim=0)
        else: # skip irrelevant test samples
            pass

    intra_var = calc_intravar(feat_cls) # d(var)/d(theta)
    params = model.module[-1].weight # last linear layer
    grad_intravar2params = list(grad(intra_var, params))
    return grad_intravar2params


def inverse_hessian_product(model, criterion, v, dl_tr,
                            scale=500, damping=0.01):
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
            damping: float, dampening factor "chosen to be roughly the size of the most negative eigenvalue of the empirical Hessian (so that it becomes PSD)."
            scale: float, scaling factor, "the scale parameter scales the maximum eigenvalue to < 1 so that our Taylor approximation converges, otherwise h_estimate get NaN"
        Returns:
            h_estimate: list of torch tensors, s_test
    """
    cur_estimate = v.copy() # current estimate
    for _, (x, t, _) in tqdm(enumerate(dl_tr)): # I change it to be looping over all training samples
        x, t = x.cuda(), t.cuda()
        model.eval(); model.zero_grad()
        criterion.zero_grad(); criterion.proxies.requires_grad = False
        m = model(x)
        loss = criterion(m, None, t)
        params = [model.module[-1].weight]
        hv = hessian_vector_product(loss, params, cur_estimate) # get hvp
        # Inverse Hessian product Update: v + (I - Hessian_at_x) * cur_estimate
        cur_estimate = [_v + (1 - damping) * _h_e - _hv / scale for _v, _h_e, _hv in zip(v, cur_estimate, hv)]

    inverse_hvp = [b.detach().cpu()/scale for b in cur_estimate] # "In the loop, we scale the Hessian down by scale, which means that the estimate of the inverse Hessian-vector product will be scaled up by scale. The last division corrects for this scaling."
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
    first_grads = grad(y, x, retain_graph=True, create_graph=True)
    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem.detach()) # v is considered as constant
    # Second backprop
    return_grads = grad(elemwise_products, x)
    return return_grads

def calc_influential_func(inverse_hvp, grad_alltrain):
    '''
        Calculate influential functions
        Arguments:
            inverse_hvp: inverse hessian vector product, of shape (|theta|,)
            grad_alltrain: list of gradients for all training (N_train, |theta|)
        Returns:
            influence_values: list of influence values (N_train,)
    '''
    influence_values = []
    for grad1train in grad_alltrain:
        # influence = (-1) * grad(test)' H^-1 grad(train), dont forget the negative sign
        influence_thistrain = [(-1)*torch.dot(x.flatten().detach().cpu(), y.flatten()).item() \
                               for x, y in zip(inverse_hvp, grad1train)]
        influence_values.append(influence_thistrain)
    return influence_values