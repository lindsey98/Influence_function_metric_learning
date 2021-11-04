"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np

import torch
from torch import nn
from torch import autograd
from blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock
import torch.nn.functional as F
import torchvision.models as models

class Generator(nn.Module):
    '''
        Generator which includes content encoder, class encoder, decoder
        Takes images as input, and output as reconstructed image
    '''
    def __init__(self, hp):
        super(Generator, self).__init__()

        n_res_blks = hp['n_res_blks']
        latent_dim = hp['latent_dim']

        self.enc = ContentEncoder(sz_embed=latent_dim)

        self.dec = Decoder(4,
                           n_res_blks,
                           2048,
                           3,
                           res_norm='bn',
                           activ='relu',
                           pad_type='reflect')

    def forward(self, one_image):
        # reconstruct an image
        content, avg_feat = self.encode(one_image)
        images_trans = self.decode(content)
        return images_trans, avg_feat

    def encode(self, one_image):
        content, avg_feat = self.enc(one_image)
        return content, avg_feat

    def decode(self, content):
        images = self.dec(content)
        return images


class ContentEncoder(nn.Module):
    def __init__(self, model='resnet50', sz_embed=512):
        super(ContentEncoder, self).__init__()
        self.base = models.__dict__[model](pretrained=True) # Resnet50 backbone
        self.pool = nn.AdaptiveMaxPool2d((1, 1))

        self.lnorm = nn.LayerNorm(2048, elementwise_affine=False).cuda()
        self.emb = torch.nn.Linear(2048, sz_embed)

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x) #

        x1 = self.base.layer1(x)  # size//2
        x2 = self.base.layer2(x1)  # size//4
        x3 = self.base.layer3(x2) # size//8
        feat = self.base.layer4(x3) # size//16

        avg_feat = self.pool(feat)
        avg_feat = avg_feat.reshape(avg_feat.size()[0], -1)
        avg_feat = self.lnorm(avg_feat)
        avg_feat = self.emb(avg_feat)  # feature map of (B, sz_embed, H, W)
        return feat, avg_feat


class Decoder(nn.Module):
    def __init__(self, ups, n_res, dim, out_dim, res_norm, activ, pad_type):
        super(Decoder, self).__init__()

        self.model = []
        self.model += [ResBlocks(n_res, dim, res_norm,
                                 activ, pad_type=pad_type)]
        for i in range(ups): # should be 4?
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2,
                                       norm='in',
                                       activation=activ,
                                       pad_type=pad_type)]
            dim //= 2
        self.model += [Conv2dBlock(dim, out_dim, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

