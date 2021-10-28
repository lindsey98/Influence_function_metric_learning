'''
From https://github.com/F-Salehi/CURE_robustness/blob/daeef24ea443c37a55f87f93679a44d8504d88e7/CURE/CURE.py#L200
'''
import torch

def _find_z(inputs, targets, model, criterion, h=3.):
    '''
    inputs: batch of images
    '''
    inputs.requires_grad_()
    outputs = model.eval()(inputs)

    loss_z = criterion(outputs, None, targets)
    loss_z.backward()

    grad = inputs.grad.data + 0.0
    norm_grad = grad.norm().item()
    z = torch.sign(grad).detach() + 0.
    z = h*(z + 1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None] + 1e-7) # of shape same as inputs

    inputs.grad.detach_()
    inputs.grad.zero_()
    model.zero_grad()

    model.train()
    return z, norm_grad


def regularizer(inputs, targets, model, criterion, lambda_=4.):
    '''
    Regularizer term in CURE
    '''
    z, norm_grad = _find_z(inputs, targets, model, criterion)

    inputs.requires_grad_()
    outputs_pos = model.eval()(inputs + z)
    outputs_orig = model.eval()(inputs)

    loss_pos = criterion(outputs_pos, None, targets)
    loss_orig = criterion(outputs_orig, None, targets)
    grad_diff = \
        torch.autograd.grad((loss_pos - loss_orig), inputs,
                            create_graph=False)[0]
    reg = grad_diff.reshape(grad_diff.size()[0], -1).norm(dim=1)
    model.zero_grad()
    model.train()

    return torch.sum(lambda_ * reg) / float(inputs.size(0)), norm_grad
