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
from utils import predict_batchwise, inner_product_sim
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
            num_workers = 0,
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
        for ct, (x,y,_) in tqdm(enumerate(dl_tr_noshuffle)): #FIXME: memory error
            training_x = torch.cat((training_x, x), dim=0)
            training_y = torch.cat((training_y, y), dim=0)
        torch.save(training_x, os.path.join(root, 'Training_data', 'training_dataset_data.pth'))
        torch.save(training_y, os.path.join(root, 'Training_data', 'training_dataset_label.pth'))

        test_x, test_y = torch.tensor([]), torch.tensor([])
        for ct, (x,y,_) in tqdm(enumerate(dl_ev)):
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



if __name__ == '__main__':

    dataset_name = 'logo2k'
    dynamic_proxy = True
    sz_embedding = 2048
    presaved = True
    pretrained = True

    os.makedirs('dvi_data_{}_{}/'.format(dataset_name, dynamic_proxy), exist_ok=True)
    os.makedirs(os.path.join('dvi_data_{}_{}/'.format(dataset_name, dynamic_proxy), 'Training_data'), exist_ok=True)
    os.makedirs(os.path.join('dvi_data_{}_{}/'.format(dataset_name, dynamic_proxy), 'Testing_data'), exist_ok=True)
    os.makedirs('dvi_data_{}_{}/resnet_{}_umap_plots/'.format(dataset_name, dynamic_proxy, sz_embedding), exist_ok=True)

    model_dir = 'dvi_data_{}_{}/ResNet_{}_Model'.format(dataset_name, dynamic_proxy, sz_embedding)
    plot_dir = 'dvi_data_{}_{}/resnet_{}_umap_plots'.format(dataset_name, dynamic_proxy, sz_embedding)

    os.makedirs(plot_dir, exist_ok=True)
    dl_tr, dl_ev = prepare_data(data_name=dataset_name, root='dvi_data_{}_{}/'.format(dataset_name, dynamic_proxy), save=False)

    # exit()

    # load model
    feat = Feat_resnet50_max_n()
    in_sz = feat(torch.rand(1, 3, 256, 256)).squeeze().size(0)
    feat.train()
    emb = torch.nn.Linear(in_sz, sz_embedding)
    model = torch.nn.Sequential(feat, emb)
    model = nn.DataParallel(model)
    model.cuda()

    # load loss
    # FIXME: should save the whole criterion class because mask might be updating
    criterion = ProxyNCA_prob(nb_classes = dl_tr.dataset.nb_classes(),
                              sz_embed = sz_embedding,
                              scale=3)

    with open("{0}/{1}_ip.json".format('log', '{}_{}_trainval_{}_0_{}'.format(dataset_name, dataset_name, sz_embedding, dynamic_proxy)), 'rt') as handle:
        cache_sim = json.load(handle)
    with open("{0}/{1}_cls.json".format('log', '{}_{}_trainval_{}_0_{}'.format(dataset_name, dataset_name, sz_embedding, dynamic_proxy)), 'rt') as handle:
        cache_label = json.load(handle)

    # Line plot which show the trend of inner_prod_sim to nearest ground-truth class's proxy
    # os.makedirs(os.path.join(plot_dir, 'line_plot'), exist_ok=True)
    # for cls in range(criterion.nb_classes):
    #     sim_cls1 = []
    #     for e in tqdm(range(40)):
    #         sim = cache_sim[str(e)]
    #         labels = cache_label[str(e)]
    #         sim_cls = np.asarray(sim)[np.asarray(labels) == cls].tolist()
    #         sim_cls1.append(sim_cls)
    #
    #     sim_cls1 = np.asarray(sim_cls1)
    #     fig = plt.figure()
    #     for j in range(sim_cls1.shape[1]):
    #         plt.plot(range(40), sim_cls1[:, j], linestyle='-', color='k', linewidth=0.5)
    #     fig.savefig(os.path.join(plot_dir, 'line_plot', 'cls{}.png'.format(str(cls))))

    for i in range(1, 2):
        # subclasses = np.asarray(list(range(10*(i-1), 10*i)))
        subclasses = np.asarray([1, 11, 21, 23, 25, 26, 44, 46, 49, 50])
        for e in tqdm([0, 9, 19, 20, 29, 30, 39]):

            model.load_state_dict(torch.load('{}/Epoch_{}/{}_{}_trainval_2048_0.pth'.format(model_dir, e+1, dataset_name, dataset_name)))
            proxies = torch.load('{}/Epoch_{}/proxy.pth'.format(model_dir, e+1), map_location='cpu')['proxies'].detach()
            proxies = proxies.view(criterion.nb_classes, criterion.max_proxy_per_class, -1)
            mask = torch.load('{}/Epoch_{}/proxy.pth'.format(model_dir, e+1), map_location='cpu')['mask'].detach()
            count_proxy = torch.sum(mask, -1).detach().cpu().numpy().tolist()
            used_proxies = []
            for m, n in enumerate(count_proxy):
                used_proxies.append(proxies[m, :int(n)]) # of shape (C, sz_embedding)
            stacked_proxies = torch.cat(used_proxies, dim=0)

            if not presaved:
                embedding, label, *_ = predict_batchwise(model, dl_tr)
                torch.save(embedding, '{}/Epoch_{}/training_embeddings.pth'.format(model_dir, e+1))
                torch.save(label, '{}/Epoch_{}/training_labels.pth'.format(model_dir, e+1))
            else:
                embedding = torch.load('{}/Epoch_{}/training_embeddings.pth'.format(model_dir, e+1))
                label = torch.load('{}/Epoch_{}/training_labels.pth'.format(model_dir, e+1))
            embedding, stacked_proxies = F.normalize(embedding, dim=-1), F.normalize(stacked_proxies, dim=-1) # need to normalize, other producing wierd results
            print(embedding.shape, stacked_proxies.shape)

            # Parametric Umap model
            encoder = encoder_model()
            embedder = ParametricUMAP(encoder=encoder, verbose=False, batch_size=256)

            if not pretrained:
                if e > 0:
                    try:
                        # Initialize by last visualization model
                        embedder.encoder = tf.keras.models.load_model('{}/Epoch_{}/parametric_model/encoder'.format(model_dir, e))
                    except OSError as error: # saved model file does not exist
                        print(error)
                        pass
                # Train on all samples and all proxies
                embedder.fit_transform(np.concatenate((embedding.detach().cpu().numpy(), stacked_proxies.cpu().numpy()), axis=0))
                embedder.encoder.save('{}/Epoch_{}/parametric_model/encoder'.format(model_dir, e+1))
            embedder.encoder = tf.keras.models.load_model('{}/Epoch_{}/parametric_model/encoder'.format(model_dir, e+1))

            # transform high dimensional embedding and proxy to low-dimension
            low_dim_emb = embedder.transform(embedding.detach().cpu().numpy())
            low_dim_proxy = []
            for p in used_proxies:
                p = F.normalize(p, p=2, dim=-1)
                low_dim_proxy.append(embedder.transform(p.cpu().numpy()))
            print(low_dim_emb.shape)
            print(len(low_dim_proxy))

            # Only visualize subset of 10 classes
            label_sub = label[np.isin(label, subclasses)].numpy()
            label_cmap = {v: k for k, v in enumerate(subclasses)}
            print(label_cmap)
            low_dim_emb = low_dim_emb[np.isin(label, subclasses), :]
            low_dim_proxy_sub = []
            low_dim_proxy_labels = []
            for m, p in enumerate(low_dim_proxy):
                if m in subclasses:
                    for sub_p in low_dim_proxy[m]:
                        low_dim_proxy_labels.append(m)
                        low_dim_proxy_sub.append(sub_p)
            low_dim_proxy_sub = np.asarray(low_dim_proxy_sub)
            print(low_dim_proxy_sub.shape)

            # Visualize
            classes = subclasses.tolist()
            fig, ax = plt.subplots()
            x = low_dim_emb[:, 0]
            y = low_dim_emb[:, 1]
            line = ax.scatter(x, y,c=[label_cmap[x] for x in label_sub],
                                   cmap='tab10', s=5)
            ax.scatter(low_dim_proxy_sub[:, 0], low_dim_proxy_sub[:, 1],
                        c=[label_cmap[x] for x in low_dim_proxy_labels],
                        cmap='tab10', marker=(5,1), edgecolors='black')
            plt.legend(handles=ax.legend_elements()[0], labels=classes)

            imagebox = OffsetImage(dl_tr.dataset.__getitem__(0)[0].permute(1, 2, 0).numpy(), zoom=0.2)
            xybox = (50., 50.)
            ab = AnnotationBbox(imagebox, (0, 0),
                                xybox=xybox,
                                xycoords='data',
                                boxcoords="offset points",
                                pad=0.3, arrowprops=dict(arrowstyle="->")
                                )
            ax.add_artist(ab)
            ab.set_visible(False)

            def hover(event):
                # if the mouse is over the scatter points
                if line.contains(event)[0]:
                    # find out the index within the array from the event
                    ind, = line.contains(event)[1]["ind"]
                    # get the figure size
                    w, h = fig.get_size_inches() * fig.dpi
                    ws = (event.x > w / 2.) * -1 + (event.x <= w / 2.)
                    hs = (event.y > h / 2.) * -1 + (event.y <= h / 2.)
                    # if event occurs in the top or right quadrant of the figure,
                    # change the annotation box position relative to mouse.
                    ab.xybox = (xybox[0]*ws, xybox[1]*hs)
                    # make annotation box visible
                    ab.set_visible(True)
                    # place it at the position of the hovered scatter point
                    ab.xy = (x[ind], y[ind])
                    # set the image corresponding to that point
                    imagebox.set_data(dl_tr.dataset.__getitem__(ind)[0].permute(1, 2, 0).numpy())
                else:
                    # if the mouse is not over a scatter point
                    ab.set_visible(False)
                fig.canvas.draw_idle()

            # add callback for mouse moves
            fig.canvas.mpl_connect('motion_notify_event', hover)

            plt.draw()
            plt.show()
            os.makedirs('{}/{}th_batch'.format(plot_dir, str(i)), exist_ok=True)

    #         # fig.savefig('{}/{}th_batch/Epoch_{}.png'.format(plot_dir, str(i), e+1))
    #         # break