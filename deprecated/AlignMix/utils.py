"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import yaml
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils

import sklearn.cluster
import sklearn.metrics.cluster
import numpy as np
import math
from tqdm import tqdm
import logging


def update_average(model_tgt, model_src, beta=0.999):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert(p_src is not p_tgt)
            p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def make_result_folders(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def __write_images(im_outs, dis_img_n, file_name):
    im_outs = [images.expand(-1, 3, -1, -1) for images in im_outs]
    image_tensor = torch.cat([images[:dis_img_n] for images in im_outs], 0)
    image_grid = vutils.make_grid(image_tensor.data,
                                  nrow=dis_img_n, padding=0, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1)


def write_1images(image_outputs, image_directory, postfix):
    display_image_num = image_outputs[0].size(0)
    __write_images(image_outputs, display_image_num,
                   '%s/gen_%s.jpg' % (image_directory, postfix))


def _write_row(html_file, it, fn, all_size):
    html_file.write("<h3>iteration [%d] (%s)</h3>" % (it, fn.split('/')[-1]))
    html_file.write("""
        <p><a href="%s">
          <img src="%s" style="width:%dpx">
        </a><br>
        <p>
        """ % (fn, fn, all_size))
    return


def write_html(filename, it, img_save_it, img_dir, all_size=1536):
    html_file = open(filename, "w")
    html_file.write('''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Experiment name = %s</title>
      <meta http-equiv="refresh" content="30">
    </head>
    <body>
    ''' % os.path.basename(filename))
    html_file.write("<h3>current</h3>")
    _write_row(html_file, it, '%s/gen_train_current.jpg' % img_dir, all_size)
    for j in range(it, img_save_it - 1, -1):
        _write_row(html_file, j, '%s/gen_train_%08d.jpg' % (img_dir, j),
                   all_size)
    html_file.write("</body></html>")
    html_file.close()


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer)
               if ((not callable(getattr(trainer, attr))
                    and not attr.startswith("__"))
                    and ('loss' in attr
                        or 'grad' in attr
                        or 'nwd' in attr
                        or 'accuracy' in attr))]
    print(members)
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


def cluster_by_kmeans(X, nb_clusters):
    """
    xs : embeddings with shape [nb_samples, nb_features]
    nb_clusters : in this case, must be equal to number of classes
    """
    return sklearn.cluster.KMeans(nb_clusters).fit(X).labels_

def calc_normalized_mutual_information(ys, xs_clustered):
    return sklearn.metrics.cluster.normalized_mutual_info_score(xs_clustered, ys, average_method='geometric')

def assign_by_euclidian_at_k(X, T, k):
    """
        X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
        k : for each sample, assign target labels of k nearest points
    """
    # distances = sklearn.metrics.pairwise.pairwise_distances(X)
    chunk_size = 1000
    num_chunks = math.ceil(len(X)/chunk_size)
    distances = torch.tensor([])
    for i in tqdm(range(0, num_chunks)):
        chunk_indices = [chunk_size*i, min(len(X), chunk_size*(i+1))]
        chunk_X = X[chunk_indices[0]:chunk_indices[1], :]
        distance_mat = torch.from_numpy(sklearn.metrics.pairwise.pairwise_distances(X, chunk_X))
        distances = torch.cat((distances, distance_mat), dim=-1)
    assert distances.shape[0] == len(X)
    assert distances.shape[1] == len(X)

    distances = distances.numpy()
    # get nearest points
    indices   = np.argsort(distances, axis = 1)[:, 1 : k + 1]

    return np.array([[T[i] for i in ii] for ii in indices])


def assign_by_euclidian_at_k_indices(X, T, k):
    """
        X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
        k : for each sample, assign target labels of k nearest points
    """
    # distances = sklearn.metrics.pairwise.pairwise_distances(X)
    chunk_size = 1000
    num_chunks = math.ceil(len(X)/chunk_size)
    distances = torch.tensor([])
    for i in tqdm(range(0, num_chunks)):
        chunk_indices = [chunk_size*i, min(len(X), chunk_size*(i+1))]
        chunk_X = X[chunk_indices[0]:chunk_indices[1], :]
        distance_mat = torch.from_numpy(sklearn.metrics.pairwise.pairwise_distances(X, chunk_X))
        distances = torch.cat((distances, distance_mat), dim=-1)
    assert distances.shape[0] == len(X)
    assert distances.shape[1] == len(X)

    distances = distances.numpy()
    # get nearest points
    indices   = np.argsort(distances, axis = 1)[:, 1 : k + 1]

    return indices, np.array([[T[i] for i in ii] for ii in indices])


def calc_recall_at_k(T, Y, k):
    """
        Check whether a sample's KNN contain any sample with the same class labels as itself
        T : [nb_samples] (target labels)
        Y : [nb_samples x k] (k predicted labels/neighbours)
    """
    s = sum([1 for t, y in zip(T, Y) if t in y[:k]])
    return s / (1. * len(T))






