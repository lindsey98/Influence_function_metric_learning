from torch import nn
import torchvision.models as models
import torch
import torch.nn.functional as F
from similarity import pairwise_distance
from loss import binarize_and_smooth_labels
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    '''
        Feature embedding network
    '''
    def __init__(self, nb_classes, sz_embed=512, scale=3, model='resnet50', use_lnorm=True):
        nn.Module.__init__(self)
        self.model = model

        self.base = models.__dict__[model](pretrained=True)
        self.lnorm = None
        if use_lnorm:
            self.lnorm = nn.LayerNorm(sz_embed, elementwise_affine=False).cuda()

        self.emb = nn.Conv2d(2048, sz_embed, 1, 1)

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed) / 8)
        self.scale = scale

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x1 = self.base.layer1(x)
        x2 = self.base.layer2(x1)
        x3 = self.base.layer3(x2)
        x4 = self.base.layer4(x3)
        feat = self.emb(x4) # B, sz_embed, H, W

        c_ij = self.dynamic_routing(feat) # find best pooling weights for each specific sample
        final_feat = (c_ij * feat).sum(-1).sum(-1) # dynamic pooling

        if self.lnorm is not None:
            final_feat = self.lnorm(final_feat)
        return final_feat

    def dynamic_routing(self, feat):
        HW = feat.size()[2]*feat.size()[3]
        b_ij = torch.autograd.Variable(torch.ones(feat.size()[0], 1, feat.size()[2], feat.size()[3]) / HW ) # B, 1, H, W
        b_ij = b_ij.to(feat.device)

        num_iterations = 1
        for iteration in range(num_iterations):
            weights_sum = torch.sum(torch.exp(b_ij), dim=(-1, -2), keepdim=True)
            c_ij = torch.exp(b_ij) / weights_sum

            s_j = torch.sum(c_ij * feat, dim=(-1, -2)) # B, sz_embed
            squash_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = feat * squash_j.unsqueeze(-1).unsqueeze(-1)
                b_ij = b_ij + a_ij
                # plt.subplot(1,2,1)
                # plt.imshow(image[0].permute(1,2,0).detach().cpu().numpy())
                # plt.subplot(1,2,2)
                # plt.imshow(c_ij[0, 0, :, :].detach().cpu().numpy())
                # plt.colorbar()
                # plt.show()

        return c_ij

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

    def calc_loss(self, imageX, T):

        P = self.proxies
        X = self.forward(imageX)

        P = self.scale * F.normalize(P, p=2, dim=-1)
        X = self.scale * F.normalize(X, p=2, dim=-1)

        D = pairwise_distance(
            torch.cat(
                [X, P]
            ),
            squared=True
        )[0][:X.size()[0], X.size()[0]:]

        T = binarize_and_smooth_labels(
            T=T, nb_classes=len(P), smoothing_const=0
        )

        loss = torch.sum(- T * F.log_softmax(-D, -1), -1)
        loss = loss.mean()
        return loss
