import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import utils
import dataset
from tqdm import tqdm
from networks import bninception, Feat_resnet50_max_n
from torch import nn
import umap
from umap.parametric_umap import ParametricUMAP
import tensorflow as tf
from umap.parametric_umap import load_ParametricUMAP
from loss import ProxyNCA_prob
from utils import predict_batchwise, inner_product_sim, predict_batchwise_loss
import json
import torch.nn.functional as F
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, \
    AnnotationBbox

os.environ['CUDA_VISIBLE_DEVICES'] = "1,0"


def prepare_data(data_name='cub', root='dvi_data_cub200/', save=False):
    '''
        Prepare dataloader
        :param data_name: dataset used
        :param root:root dir to save data
        :param save: if save is True, save data
    '''
    dataset_config = utils.load_config('dataset/config.json')

    config = utils.load_config('config/{}.json'.format(data_name))
    transform_key = 'transform_parameters'
    if 'transform_key' in config.keys():
        transform_key = config['transform_key']
    print('Transformation: ', transform_key)

    dl_tr_noshuffle = torch.utils.data.DataLoader(
        dataset=dataset.load(
            name=data_name,
            root=dataset_config['dataset'][data_name]['root'],
            source=dataset_config['dataset'][data_name]['source'],
            classes=dataset_config['dataset'][data_name]['classes']['trainval'],
            transform=dataset.utils.make_transform(
                **dataset_config[transform_key],
                is_train=False
            )
        ),
        num_workers=0,
        shuffle=False,
        batch_size=64,
    )

    dl_ev = torch.utils.data.DataLoader(
        dataset.load(
            name=data_name,
            root=dataset_config['dataset'][data_name]['root'],
            source=dataset_config['dataset'][data_name]['source'],
            classes=dataset_config['dataset'][data_name]['classes']['eval'],
            transform=dataset.utils.make_transform(
                **dataset_config[transform_key],
                is_train=False
            )
        ),
        batch_size=128,
        shuffle=False,
        num_workers=0,

    )

    if save:
        training_x, training_y = torch.tensor([]), torch.tensor([])
        for ct, (x, y, _) in tqdm(enumerate(dl_tr_noshuffle)):  # FIXME: memory error
            training_x = torch.cat((training_x, x), dim=0)
            training_y = torch.cat((training_y, y), dim=0)
        torch.save(training_x, os.path.join(root, 'Training_data', 'training_dataset_data.pth'))
        torch.save(training_y, os.path.join(root, 'Training_data', 'training_dataset_label.pth'))

        test_x, test_y = torch.tensor([]), torch.tensor([])
        for ct, (x, y, _) in tqdm(enumerate(dl_ev)):
            test_x = torch.cat((test_x, x), dim=0)
            test_y = torch.cat((test_y, y), dim=0)
        torch.save(test_x, os.path.join(root, 'Testing_data', 'testing_dataset_data.pth'))
        torch.save(test_y, os.path.join(root, 'Testing_data', 'testing_dataset_label.pth'))

    return dl_tr_noshuffle, dl_ev


def encoder_model(n_components=2):
    '''
        Customized encoder
        :param n_components: low dimensional projection dimensions
    '''
    encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(units=256, activation="relu"),
        tf.keras.layers.Dense(units=256, activation="relu"),
        tf.keras.layers.Dense(units=n_components),
    ])
    return encoder


def pumap_training(model, model_dir, e,
                   criterion,
                   dl_tr, dl_ev,
                   stacked_proxies, presaved, pretrained):
    '''
        Parameteric umap training
    '''
    if not presaved:
        embedding, label, indices = predict_batchwise(model, dl_tr)
        torch.save(embedding, '{}/Epoch_{}/training_embeddings.pth'.format(model_dir, e + 1))
        torch.save(label, '{}/Epoch_{}/training_labels.pth'.format(model_dir, e + 1))
        torch.save(indices, '{}/Epoch_{}/training_indices.pth'.format(model_dir, e + 1))

        testing_embedding, testing_label, testing_indices = predict_batchwise(model, dl_ev)
        torch.save(embedding, '{}/Epoch_{}/testing_embeddings.pth'.format(model_dir, e + 1))
        torch.save(label, '{}/Epoch_{}/testing_labels.pth'.format(model_dir, e + 1))
        torch.save(indices, '{}/Epoch_{}/testing_indices.pth'.format(model_dir, e + 1))

    else:
        embedding = torch.load('{}/Epoch_{}/training_embeddings.pth'.format(model_dir, e + 1))
        label = torch.load('{}/Epoch_{}/training_labels.pth'.format(model_dir, e + 1))
        indices = torch.load('{}/Epoch_{}/indices.pth'.format(model_dir, e + 1))

        testing_embedding = torch.load('{}/Epoch_{}/testing_embeddings.pth'.format(model_dir, e + 1))
        testing_label = torch.load('{}/Epoch_{}/testing_labels.pth'.format(model_dir, e + 1))
        testing_indices = torch.load('{}/Epoch_{}/testing_indices.pth'.format(model_dir, e + 1))

    # need to normalize, other producing wierd results
    embedding, stacked_proxies = F.normalize(embedding, dim=-1), F.normalize(stacked_proxies, dim=-1)
    testing_embedding = F.normalize(testing_embedding, dim=-1)
    print('Embedding of shape: ', embedding.shape,
          'Current proxies of shape: ', stacked_proxies.shape,
          'Testing embedding of shape: ', testing_embedding.shape)

    # Parametric Umap model
    encoder = encoder_model()
    embedder = ParametricUMAP(encoder=encoder, verbose=False, batch_size=256)

    if not pretrained:
        if e > 0:
            try:
                # Initialize by last visualization model
                embedder.encoder = tf.keras.models.load_model(
                    '{}/Epoch_{}/parametric_model/encoder'.format(model_dir, e))
            except OSError as error:  # saved model file does not exist
                print(error)
                pass
        # Train on all samples and all proxies
        embedder.fit_transform(np.concatenate((embedding.detach().cpu().numpy(),
                                               stacked_proxies.cpu().numpy(),
                                               testing_embedding.cpu().numpy()), axis=0))
        embedder.encoder.save('{}/Epoch_{}/parametric_model/encoder'.format(model_dir, e + 1))
    else:
        embedder.encoder = tf.keras.models.load_model('{}/Epoch_{}/parametric_model/encoder'.format(model_dir, e + 1))

    # transform high dimensional embedding and proxy to low-dimension
    low_dim_emb = embedder.transform(embedding.detach().cpu().numpy())
    low_dim_emb_test = embedder.transform(testing_embedding.detach().cpu().numpy())
    low_dim_proxy = embedder.transform(p.cpu().numpy())

    return embedder, \
            low_dim_proxy,\
           (low_dim_emb, label, indices),\
           (low_dim_emb_test, testing_label, testing_indices)


if __name__ == '__main__':

    dataset_name = 'logo2k'
    sz_embedding = 512
    presaved = True
    pretrained = True
    interactive = False

    folder = 'dvi_data_{}_{}/'.format(dataset_name)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, 'Training_data'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'Testing_data'), exist_ok=True)
    os.makedirs('{}/resnet_{}_umap_plots/'.format(folder, sz_embedding), exist_ok=True)

    model_dir = '{}/ResNet_{}_Model'.format(folder, sz_embedding)
    plot_dir = '{}/resnet_{}_umap_plots'.format(folder, sz_embedding)

    os.makedirs(plot_dir, exist_ok=True)
    dl_tr, dl_ev = prepare_data(data_name=dataset_name, root=folder, save=False)

    # load model
    feat = Feat_resnet50_max_n()
    in_sz = feat(torch.rand(1, 3, 256, 256)).squeeze().size(0)
    feat.train()
    emb = torch.nn.Linear(in_sz, sz_embedding)
    model = torch.nn.Sequential(feat, emb)
    model = nn.DataParallel(model)
    model.cuda()

    # load loss
    # TODO: should echange criterion
    criterion = ProxyNCA_prob(nb_classes=dl_tr.dataset.nb_classes(),
                              sz_embed=sz_embedding,
                              scale=3)

    # for i in range(1, 10):
    #     subclasses = np.asarray(list(range(5 * (i - 1), 5 * i)))
    # for e in tqdm(range(39)):
    for e in [39]:
        model.load_state_dict(torch.load(
            '{}/Epoch_{}/{}_{}_trainval_512_0.pth'.format(model_dir, e + 1, dataset_name, dataset_name)))
        proxies = torch.load('{}/Epoch_{}/proxy.pth'.format(model_dir, e + 1), map_location='cpu')['proxies'].detach()

        # TODO: reload criterion
        criterion.proxies.data = proxies
        embedder, low_dim_proxy, (low_dim_emb, label, _), (low_dim_emb_test, test_label, _) = pumap_training(model=model,
                                                                                       model_dir=model_dir,
                                                                                       e=e,
                                                                                       criterion=criterion,
                                                                                       stacked_proxies=proxies,
                                                                                       dl_tr=dl_tr,
                                                                                       dl_ev=dl_ev,
                                                                                       presaved=presaved,
                                                                                       pretrained=pretrained)
        '''Visualize'''

        # Training
        indices = range(len(low_dim_emb))
        images = [dl_tr.dataset.__getitem__(ind)[0].permute(1, 2, 0).numpy() for ind in indices]
        label_sub = label[indices].numpy()
        low_dim_emb = low_dim_emb[indices, :]

        # Testing
        indices_test = range(len(low_dim_emb_test))
        images_test = [dl_ev.dataset.__getitem__(ind)[0].permute(1, 2, 0).numpy() for ind in indices_test]
        label_sub_test = test_label[indices].numpy()
        low_dim_emb_test = low_dim_emb_test[indices, :]

        # # Visualize
        fig, ax = plt.subplots()
        # For embedding points
        x, y = low_dim_emb[:, 0], low_dim_emb[:, 1]
        x_test, y_test = low_dim_emb_test[:, 0], low_dim_emb_test[:, 1]
        px, py = low_dim_proxy[:, 0], low_dim_proxy[:, 1]
        ax.set_xlim(min(min(x), min(px)), max(max(x), max(px))); ax.set_ylim(min(min(y), min(py)), max(max(y), max(py)))

        line4tr = ax.scatter(x, y, c='gray',  s=5)
        line4proxy = ax.scatter(px, py, c='blue',  marker=(5, 1), edgecolors='black')
        line4ev = ax.scatter(x_test, y_test, c='pink', s=5)

        if interactive:
            imagebox = OffsetImage(dl_tr.dataset.__getitem__(0)[0].permute(1, 2, 0).numpy(), zoom=0.2)
            xybox = (32., 32.)
            ab = AnnotationBbox(imagebox, (0, 0),
                                xybox=xybox,
                                xycoords='data',
                                boxcoords="offset points",
                                arrowprops=dict(arrowstyle="->"))
            ax.add_artist(ab)
            ab.set_visible(False)

            imagebox2 = OffsetImage(dl_ev.dataset.__getitem__(0)[0].permute(1, 2, 0).numpy(), zoom=0.2)
            ab2 = AnnotationBbox(imagebox2, (0, 0),
                                xybox=xybox,
                                xycoords='data',
                                boxcoords="offset points",
                                arrowprops=dict(arrowstyle="->"))
            ax.add_artist(ab2)
            ab2.set_visible(False)

            xybox_ac = (50., 50.)
            ac = ax.annotate("", xy=(0, 0),
                             xytext=xybox_ac, textcoords="offset points",
                             bbox=dict(boxstyle='round4', fc='linen', ec='k', lw=1),
                             arrowprops=dict(arrowstyle='->'))
            ax.add_artist(ac)
            ac.set_visible(False)

            def hover(event):

                '''
                For training
                '''

                if line4tr.contains(event)[0]:
                    # find out the index within the array from the event
                    ind, = line4tr.contains(event)[1]["ind"]
                    # get the figure size
                    w, h = fig.get_size_inches() * fig.dpi
                    ws = (event.x > w / 2.) * -1 + (event.x <= w / 2.)
                    hs = (event.y > h / 2.) * -1 + (event.y <= h / 2.)
                    # if event occurs in the top or right quadrant of the figure,
                    # change the annotation box position relative to mouse.
                    ab.xybox = (xybox[0] * ws, xybox[1] * hs)
                    ab.set_visible(True)
                    # place it at the position of the hovered scatter point
                    ab.xy = (x[ind], y[ind])
                    # set the image corresponding to that point
                    imagebox.set_data(images[ind])

                    ac.xybox = (xybox_ac[0] * ws, xybox_ac[1] * hs)
                    ac.xy = (x[ind], y[ind])
                    text = "Indices={} \n Label={}".format(indices[ind], label_sub[ind])
                    ac.set_visible(True)
                    ac.set_text(text)

                else:
                    # if the mouse is not over a scatter point
                    ab.set_visible(False)
                    ac.set_visible(False)
                fig.canvas.draw_idle()

                '''
                For testing
                '''
                if line4ev.contains(event)[0]:
                    # find out the index within the array from the event
                    ind, = line4ev.contains(event)[1]["ind"]
                    # get the figure size
                    w, h = fig.get_size_inches() * fig.dpi
                    ws = (event.x > w / 2.) * -1 + (event.x <= w / 2.)
                    hs = (event.y > h / 2.) * -1 + (event.y <= h / 2.)
                    # if event occurs in the top or right quadrant of the figure,
                    # change the annotation box position relative to mouse.
                    ab2.xybox = (xybox[0] * ws, xybox[1] * hs)
                    ab2.set_visible(True)
                    # place it at the position of the hovered scatter point
                    ab2.xy = (x_test[ind], y_test[ind])
                    # set the image corresponding to that point
                    imagebox2.set_data(images_test[ind])

                    ac.xybox = (xybox_ac[0] * ws, xybox_ac[1] * hs)
                    ac.xy = (x_test[ind], y_test[ind])
                    text = "Indices={} \n Label={}".format(indices_test[ind], label_sub_test[ind])
                    ac.set_visible(True)
                    ac.set_text(text)

                else:
                    # if the mouse is not over a scatter point
                    ab.set_visible(False)
                    ac.set_visible(False)
                fig.canvas.draw_idle()

                '''
                For Proxy
                '''
                if line4proxy.contains(event)[0]:
                    # find out the index within the array from the event
                    ind, = line4proxy.contains(event)[1]["ind"]
                    w, h = fig.get_size_inches() * fig.dpi
                    ws = (event.x > w / 2.) * -1 + (event.x <= w / 2.)
                    hs = (event.y > h / 2.) * -1 + (event.y <= h / 2.)
                    ac.xybox = (xybox[0] * ws, xybox[1] * hs)
                    ac.xy = (px[ind], py[ind])
                    text = "Proxy for class: {}".format(ind)
                    ac.set_visible(True)
                    ac.set_text(text)
                else:
                    ac.set_visible(False)
                fig.canvas.draw_idle()


            # # add callback for mouse moves
            fig.canvas.mpl_connect('motion_notify_event', hover)
            plt.draw()
            plt.show()

        # else:
            # os.makedirs('{}/{}th_batch'.format(plot_dir, str(i)), exist_ok=True)
            # fig.savefig('{}/{}th_batch/Epoch_{}.png'.format(plot_dir, str(i), e + 1))


