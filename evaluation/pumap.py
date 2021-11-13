import matplotlib.pyplot as plt
import os

import torch

import utils
import dataset
from tqdm import tqdm
from networks import  Feat_resnet50_max_n
from torch import nn
from umap.parametric_umap import ParametricUMAP
import tensorflow as tf
from loss import *
from utils import predict_batchwise
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, \
    AnnotationBbox
import evaluation

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

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
        tf.keras.layers.InputLayer(input_shape=(512)),
        tf.keras.layers.Dense(units=256, activation="relu"),
        tf.keras.layers.Dense(units=256, activation="relu"),
        tf.keras.layers.Dense(units=n_components),
    ])
    return encoder


def pumap_training(model, model_dir, e,
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
        torch.save(testing_embedding, '{}/Epoch_{}/testing_embeddings.pth'.format(model_dir, e + 1))
        torch.save(testing_label, '{}/Epoch_{}/testing_labels.pth'.format(model_dir, e + 1))
        torch.save(testing_indices, '{}/Epoch_{}/testing_indices.pth'.format(model_dir, e + 1))

    else:
        embedding = torch.load('{}/Epoch_{}/training_embeddings.pth'.format(model_dir, e + 1))
        label = torch.load('{}/Epoch_{}/training_labels.pth'.format(model_dir, e + 1))
        indices = torch.load('{}/Epoch_{}/training_indices.pth'.format(model_dir, e + 1))

        testing_embedding = torch.load('{}/Epoch_{}/testing_embeddings.pth'.format(model_dir, e + 1))
        testing_label = torch.load('{}/Epoch_{}/testing_labels.pth'.format(model_dir, e + 1))
        testing_indices = torch.load('{}/Epoch_{}/testing_indices.pth'.format(model_dir, e + 1))

    # need to normalize, other producing wierd results
    embedding = F.normalize(embedding, dim=-1)
    stacked_proxies =  F.normalize(stacked_proxies, dim=-1)
    testing_embedding = F.normalize(testing_embedding, dim=-1)
    print('Embedding of shape: ', embedding.shape,
          'Current proxies of shape: ', stacked_proxies.shape,
          'Testing embedding of shape: ', testing_embedding.shape)

    # Parametric Umap model
    encoder = encoder_model()
    embedder = ParametricUMAP(encoder=encoder,
                              dims=(512,),
                              verbose=False, batch_size=64)

    if not pretrained:
        if e > 0:
            try:
                # Initialize by last visualization model
                embedder.encoder = tf.keras.models.load_model('{}/Epoch_{}/parametric_model/encoder'.format(model_dir, e))
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
    low_dim_proxy = embedder.transform(stacked_proxies.cpu().numpy())

    return embedder, \
           low_dim_proxy,\
           (low_dim_emb, label, indices),\
           (low_dim_emb_test, testing_label, testing_indices)

def get_wrong_indices(X, T):
    k = 1
    Y = evaluation.assign_by_euclidian_at_k(X, T, k)
    Y = torch.from_numpy(Y)
    correct = [1 if t in y[:k] else 0 for t, y in zip(T, Y)]
    wrong_ind = np.where(np.asarray(correct) == 0)[0]
    wrong_labels = T[wrong_ind]
    unique_labels, wrong_freq = torch.unique(wrong_labels, return_counts=True)
    top10_wrong_classes = unique_labels[torch.argsort(wrong_freq, descending=True)[:15]].numpy()
    return wrong_ind, top10_wrong_classes


if __name__ == '__main__':

    dataset_name = 'cub'
    # loss_type = 'ProxyNCA_prob_orig'
    loss_type = 'ProxyAnchor'
    sz_embedding = 512
    seed = 4
    presaved = False
    pretrained = False
    interactive = True
    highlight = True
    folder = 'dvi_data_{}_{}_loss{}/'.format(dataset_name, seed, loss_type)
    model_dir = '{}/ResNet_{}_Model'.format(folder, sz_embedding)
    plot_dir = '{}/resnet_{}_umap_plots'.format(folder, sz_embedding)

    os.makedirs(os.path.join(folder, 'Training_data'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'Testing_data'), exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # load data
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
    # TODO: should echange criterion accordingly
    # criterion = ProxyNCA_prob_orig(nb_classes=dl_tr.dataset.nb_classes(),
    #                               sz_embed=sz_embedding,
    #                                scale=3 )
    criterion = Proxy_Anchor(nb_classes=dl_tr.dataset.nb_classes(),
                                  sz_embed=sz_embedding)

    for e in [39]:
        model.load_state_dict(torch.load('{}/Epoch_{}/{}_{}_trainval_512_{}.pth'.format(model_dir, e + 1, dataset_name, dataset_name, seed)))
        proxies = torch.load('{}/Epoch_{}/proxy.pth'.format(model_dir, e + 1), map_location='cpu')['proxies'].detach()
        criterion.proxies.data = proxies

        embedder, low_dim_proxy, (low_dim_emb, label, _), (low_dim_emb_test, test_label, _) = pumap_training(model=model,
                                                                                       model_dir=model_dir,
                                                                                       e=e,
                                                                                       stacked_proxies=proxies,
                                                                                       dl_tr=dl_tr,
                                                                                       dl_ev=dl_ev,
                                                                                       presaved=presaved,
                                                                                       pretrained=pretrained)
        '''Visualize'''
        # Training
        indices = range(len(low_dim_emb))
        label_sub = label[indices].numpy()
        low_dim_emb = low_dim_emb[indices, :]
        images = [dl_tr.dataset.__getitem__(ind)[0].permute(1, 2, 0).numpy() for ind in indices]

        # Testing
        # indices_test = range(len(low_dim_emb_test))
        # label_sub_test = test_label[indices_test].numpy()
        # low_dim_emb_test = low_dim_emb_test[indices_test, :]

        # Wrong testing
        testing_embedding = torch.load('{}/Epoch_{}/testing_embeddings.pth'.format(model_dir, e + 1))
        testing_label = torch.load('{}/Epoch_{}/testing_labels.pth'.format(model_dir, e + 1))
        wrong_ind, top10_wrong_classes = get_wrong_indices(testing_embedding, testing_label)
        # label_sub_test_wrong = test_label[wrong_ind].numpy()
        # low_dim_emb_test_wrong = low_dim_emb_test[wrong_ind, :]
        # images_test_wrong = [dl_ev.dataset.__getitem__(ind)[0].permute(1, 2, 0).numpy() for ind in wrong_ind]
        top10_wrong_class_ind = np.where(np.isin(np.asarray(test_label), top10_wrong_classes))[0]
        top10_label_sub_test = test_label[top10_wrong_class_ind].numpy()
        top10_low_dim_emb_test = low_dim_emb_test[top10_wrong_class_ind, :]
        images_test = [dl_ev.dataset.__getitem__(ind)[0].permute(1, 2, 0).numpy() for ind in top10_wrong_class_ind]

        intersect_wrong_class_ind = np.asarray(list(set(wrong_ind).intersection(set(top10_wrong_class_ind))))
        label_sub_test_wrong = test_label[intersect_wrong_class_ind].numpy()
        low_dim_emb_test_wrong = low_dim_emb_test[intersect_wrong_class_ind, :]


        # # Visualize
        fig, ax = plt.subplots()
        # For embedding points
        x, y = low_dim_emb[:, 0], low_dim_emb[:, 1]
        x_test, y_test = top10_low_dim_emb_test[:, 0], top10_low_dim_emb_test[:, 1]
        x_test_wrong, y_test_wrong = low_dim_emb_test_wrong[:, 0], low_dim_emb_test_wrong[:, 1]
        px, py = low_dim_proxy[:, 0], low_dim_proxy[:, 1]
        ax.set_xlim(min(min(x), min(px), min(x_test)), max(max(x), max(px), max(x_test)))
        ax.set_ylim(min(min(y), min(py), min(y_test)), max(max(y), max(py), max(y_test)))

        line4tr = ax.scatter(x, y, c='gray',  s=5)
        # line4proxy = ax.scatter(px, py, c='blue', marker=(5, 1), edgecolors='black')
        line4ev = ax.scatter(x_test, y_test, c='pink', s=5)
        line4ev_wrong = ax.scatter(x_test_wrong, y_test_wrong, c='orange', s=5)


        if interactive:
            imagebox = OffsetImage(dl_tr.dataset.__getitem__(0)[0].permute(1, 2, 0).numpy(), zoom=0.2)
            xybox = (50., 50.)
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

            xybox_ac = (-50., -50.)
            ac = ax.annotate("", xy=(0, 0),
                             xytext=xybox_ac, textcoords="offset points",
                             bbox=dict(boxstyle='round4', fc='linen', ec='k', lw=1),
                             arrowprops=dict(arrowstyle='->'))
            ax.add_artist(ac)
            ac.set_visible(False)

            if highlight:
                sample_plots = []
                plot = ax.plot([], [], '.', ms=5, color='black', zorder=4)
                sample_plots.append(plot[0])
                plot[0].set_visible(False)

                sample_plots2 = []
                plot2 = ax.plot([], [], '.', ms=5, color='red', zorder=5)
                sample_plots2.append(plot2[0])
                plot2[0].set_visible(False)

                sample_dot = []
                dot = ax.plot([], [], '.', ms=5, color='blue', markeredgecolor='black', zorder=6)
                sample_dot.append(dot[0])
                dot[0].set_visible(False)

            def hover(event):
                # Training
                if line4tr.contains(event)[0]:
                    # find out the index within the array from the event
                    ind = line4tr.contains(event)[1]["ind"][0]
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
                    if highlight:
                        c = label_sub[ind]
                        data = low_dim_emb[label_sub == c, :]
                        plot[0].set_visible(True)
                        sample_plots[0].set_data(data.transpose())

                        dot[0].set_visible(True)
                        sample_dot[0].set_data((x[ind], y[ind]))

                    ac.xybox = (xybox_ac[0] * -ws, xybox_ac[1] * -hs)
                    ac.xy = (x[ind], y[ind])
                    text = "Training data indices={} \n Label={}".format(indices[ind], label_sub[ind])
                    ac.set_visible(True)
                    ac.set_text(text)
                else:
                    # if the mouse is not over a scatter point
                    ab.set_visible(False)
                    ac.set_visible(False)
                    plot[0].set_visible(False)
                    sample_dot[0].set_visible(False)
                fig.canvas.draw_idle()

                # Testing
                if line4ev.contains(event)[0]:
                    # find out the index within the array from the event
                    ind = line4ev.contains(event)[1]["ind"][0]
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
                    if highlight:
                        c = top10_label_sub_test[ind]
                        data = top10_low_dim_emb_test[top10_label_sub_test == c, :]
                        plot2[0].set_visible(True)
                        sample_plots2[0].set_data(data.transpose())

                        dot[0].set_visible(True)
                        sample_dot[0].set_data((x[ind], y[ind]))

                    ac.xybox = (xybox_ac[0] * -ws, xybox_ac[1] * -hs)
                    ac.xy = (x_test[ind], y_test[ind])
                    text = "Testing data indices={} \n Label={}".format(top10_wrong_class_ind[ind],
                                                                        top10_label_sub_test[ind])
                    ac.set_visible(True)
                    ac.set_text(text)

                else:
                    # if the mouse is not over a scatter point
                    ab2.set_visible(False)
                    ac.set_visible(False)
                    plot2[0].set_visible(False)
                    sample_dot[0].set_visible(False)
                fig.canvas.draw_idle()


            # # add callback for mouse moves
            fig.canvas.mpl_connect('motion_notify_event', hover)
            plt.draw()
            plt.show()

