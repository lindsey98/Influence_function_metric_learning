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
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"

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


def pumap_training(model, model_dir, e,
                   criterion,
                   dl_tr, stacked_proxies, presaved, pretrained):
    '''
        Parameteric Umap training
    '''
    if not presaved:
        # embedding, label, *_ = predict_batchwise(model, dl_tr)
        embedding, label, indices, gt_prob, gt_D_weighted, base_loss, p2p_sim = predict_batchwise_loss(model, dl_tr, criterion)
        torch.save(embedding, '{}/Epoch_{}/training_embeddings.pth'.format(model_dir, e + 1))
        torch.save(label, '{}/Epoch_{}/training_labels.pth'.format(model_dir, e + 1))
        torch.save(indices, '{}/Epoch_{}/indices.pth'.format(model_dir, e + 1))
        torch.save(gt_prob, '{}/Epoch_{}/gt_prob.pth'.format(model_dir, e + 1))
        torch.save(base_loss, '{}/Epoch_{}/base_loss.pth'.format(model_dir, e + 1))
        torch.save(gt_D_weighted, '{}/Epoch_{}/gt_D_weighted.pth'.format(model_dir, e + 1))
        torch.save(p2p_sim, '{}/Epoch_{}/p2p_sim.pth'.format(model_dir, e + 1))
    else:
        embedding = torch.load('{}/Epoch_{}/training_embeddings.pth'.format(model_dir, e + 1))
        label = torch.load('{}/Epoch_{}/training_labels.pth'.format(model_dir, e + 1))
        indices = torch.load('{}/Epoch_{}/indices.pth'.format(model_dir, e + 1))
        gt_prob = torch.load('{}/Epoch_{}/gt_prob.pth'.format(model_dir, e + 1))
        base_loss = torch.load('{}/Epoch_{}/base_loss.pth'.format(model_dir, e + 1))
        gt_D_weighted = torch.load('{}/Epoch_{}/gt_D_weighted.pth'.format(model_dir, e + 1))
        p2p_sim = torch.load('{}/Epoch_{}/p2p_sim.pth'.format(model_dir, e + 1))

    # need to normalize, other producing wierd results
    embedding = F.normalize(embedding, dim=-1)
    stacked_proxies = F.normalize(stacked_proxies, dim=-1)
    print('Embedding of shape: ', embedding.shape,
          'Current proxies of shape: ', stacked_proxies.shape)

    # Parametric Umap model
    encoder = encoder_model()
    embedder = ParametricUMAP(encoder=encoder, verbose=False, batch_size=64)

    if not pretrained:
        if e > 0:
            try:
                # Initialize by last visualization model
                embedder.encoder = tf.keras.models.load_model('{}/Epoch_{}/parametric_model/encoder'.format(model_dir, e))
            except OSError as error:  # saved model file does not exist
                print(error)
                pass
        # Train on all samples and all proxies
        all_data = np.concatenate((embedding.detach().cpu().numpy(), stacked_proxies.cpu().numpy()), axis=0)
        embedder.fit_transform(all_data)
        embedder.encoder.save('{}/Epoch_{}/parametric_model/encoder'.format(model_dir, e + 1))
    embedder.encoder = tf.keras.models.load_model('{}/Epoch_{}/parametric_model/encoder'.format(model_dir, e + 1))

    return embedder, embedding, label, indices, gt_prob, gt_D_weighted, base_loss, p2p_sim

def visualize_interactive(images,
                          low_dim_emb_sub, low_dim_proxy_sub,
                          label_sub, subclasses, label_cmap,
                          base_loss_sub,
                          gt_D_weighted_sub,
                          gt_prob_sub,
                          low_dim_proxy_p2psim,
                          plot_dir, i,
                          interactive=False):
    # # Visualize
    fig, ax = plt.subplots()
    # For embedding points
    x, y = low_dim_emb_sub[:, 0], low_dim_emb_sub[:, 1]
    px, py = low_dim_proxy_sub[:, 0], low_dim_proxy_sub[:, 1]

    scatter4sample = ax.scatter(x, y, c=[label_cmap[x] for x in label_sub],
                                cmap='tab10', s=5)
    plt.legend(handles=scatter4sample.legend_elements()[0], labels=subclasses.tolist())
    scatter4proxy = ax.scatter(px, py, c=[label_cmap[x] for x in low_dim_proxy_labels],
                               cmap='tab10', marker=(5, 1), edgecolors='black', linewidths=0.5)

    if interactive:
        imagebox = OffsetImage(images[0], zoom=0.2)
        xybox = (32., 32.)
        ab = AnnotationBbox(imagebox, (0, 0),
                            xybox=xybox,
                            xycoords='data',
                            boxcoords="offset points",
                            arrowprops=dict(arrowstyle="->"))
        ax.add_artist(ab)
        ab.set_visible(False)  # invisible
        xybox_ac = (50., 50.)
        ac = ax.annotate("", xy=(0, 0),
                         xytext=xybox_ac, textcoords="offset points",
                         bbox=dict(boxstyle='round4', fc='linen', ec='k', lw=1),
                         arrowprops=dict(arrowstyle='->'))
        ax.add_artist(ac)
        ac.set_visible(False)  # invisible

        xybox_ad = (-20, -20)
        ad = ax.annotate("", xy=(0, 0), xytext=xybox_ad,
                         xycoords='data', textcoords="offset points",
                         bbox=dict(boxstyle='round4', fc='linen', ec='k', lw=1),
                         arrowprops=dict(arrowstyle='->'))
        ax.add_artist(ad)
        ad.set_visible(False)  # invisible

        def hover(event):
            # if the mouse is over the scatter points
            if scatter4sample.contains(event)[0]:
                # find out the index within the array from the event
                ind, = scatter4sample.contains(event)[1]["ind"]
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
                text = "Indices={} \n Loss={:.4f} \n " \
                       "S_yi={:.4f} \n Weight2Proxy={}".format(indices[ind], base_loss_sub[ind],
                                                               gt_D_weighted_sub[ind], gt_prob_sub[ind])
                ac.set_visible(True)
                ac.set_text(text)

            else:
                # if the mouse is not over a scatter point
                ab.set_visible(False)
                ac.set_visible(False)
            fig.canvas.draw_idle()

            # if the mouse is over the scatter points
            if scatter4proxy.contains(event)[0]:
                # find out the index within the array from the event
                ind, = scatter4proxy.contains(event)[1]["ind"]
                w, h = fig.get_size_inches() * fig.dpi
                ws = (event.x > w / 2.) * -1 + (event.x <= w / 2.)
                hs = (event.y > h / 2.) * -1 + (event.y <= h / 2.)
                ad.xybox = (xybox_ad[0] * ws, xybox_ad[1] * hs)
                ad.xy = (px[ind], py[ind])
                text = "Proxy2Proxy Similarity:\n{}".format(low_dim_proxy_p2psim[ind])
                ad.set_visible(True)
                ad.set_text(text)
            else:
                ad.set_visible(False)
            fig.canvas.draw_idle()

        # # add callback for mouse moves
        fig.canvas.mpl_connect('motion_notify_event', hover)
        plt.draw()
        plt.show()

    else:
        os.makedirs('{}/{}th_batch'.format(plot_dir, str(i)), exist_ok=True)
        fig.savefig('{}/{}th_batch/Epoch_{}.png'.format(plot_dir, str(i), e + 1))


if __name__ == '__main__':

    dataset_name = 'logo2k'
    dynamic_proxy = False
    sz_embedding = 2048
    tau = 0.0
    presaved = False
    pretrained = False
    initial_proxy_num = 2
    interactive = False

    folder = 'dvi_data_{}_{}_t0.1_proxy{}_tau{}/'.format(dataset_name, dynamic_proxy, initial_proxy_num, tau)
    model_dir = '{}/ResNet_{}_Model'.format(folder, sz_embedding)
    plot_dir = '{}/resnet_{}_umap_plots'.format(folder, sz_embedding)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, 'Training_data'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'Testing_data'), exist_ok=True)
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
    # FIXME: should save the whole criterion class because mask might be updating
    criterion = ProxyNCA_prob(nb_classes = dl_tr.dataset.nb_classes(),
                              sz_embed = sz_embedding,
                              scale=3,
                              initial_proxy_num=initial_proxy_num)

    for i in range(1, 2):
        subclasses = np.asarray(list(range(5*(i-1), 5*i)))
        for e in tqdm(range(40)):
        # for e in tqdm([39]):
            model.load_state_dict(torch.load('{}/Epoch_{}/{}_{}_trainval_{}_0.pth'.format(model_dir, e+1, dataset_name, dataset_name, sz_embedding)))
            proxies = torch.load('{}/Epoch_{}/proxy.pth'.format(model_dir, e+1), map_location='cpu')['proxies'].detach()
            reshape_proxies = proxies.view(criterion.nb_classes, criterion.max_proxy_per_class, -1)
            mask = torch.load('{}/Epoch_{}/proxy.pth'.format(model_dir, e+1), map_location='cpu')['mask'].detach()
            count_proxy = torch.sum(mask, -1).detach().cpu().numpy().tolist()
            used_proxies = []
            for m, n in enumerate(count_proxy):
                used_proxies.append(reshape_proxies[m, :int(n)]) # of shape (C, sz_embedding)
            stacked_proxies = torch.cat(used_proxies, dim=0)

            #TODO: reload criterion
            criterion.proxies.data = proxies
            criterion.mask = mask
            embedder, embedding, label, _, gt_prob, gt_D_weighted, base_loss, p2p_sim = pumap_training(model=model, model_dir=model_dir, e=e,
                                                                                                       criterion=criterion,
                                                                                                       stacked_proxies=stacked_proxies, dl_tr=dl_tr,
                                                                                                       presaved=presaved, pretrained=pretrained)
            '''Visualize'''
            # transform high dimensional embedding and proxy to low-dimension
            low_dim_emb = embedder.transform(embedding.detach().cpu().numpy())
            low_dim_proxy = []
            for p in used_proxies:
                p = F.normalize(p, p=2, dim=-1)
                low_dim_proxy.append(embedder.transform(p.cpu().numpy()))

            # Only visualize subset of classes
            indices = np.where(np.isin(label, subclasses))[0]
            images = [dl_tr.dataset.__getitem__(ind)[0].permute(1, 2, 0).numpy() for ind in indices]
            label_sub = label[indices].numpy()
            gt_prob_sub = gt_prob[indices].numpy()
            base_loss_sub = base_loss[indices].numpy()
            gt_D_weighted_sub = gt_D_weighted[indices].numpy()
            low_dim_emb_sub = low_dim_emb[indices, :]

            label_cmap = {v: k for k, v in enumerate(subclasses)} # colormap
            low_dim_proxy_sub = []
            low_dim_proxy_labels = []
            low_dim_proxy_p2psim = []
            for m, p in enumerate(low_dim_proxy):
                if m in subclasses:
                    for sub_p in low_dim_proxy[m]:
                        low_dim_proxy_labels.append(m)
                        low_dim_proxy_sub.append(sub_p)
                        p2p_sim_m = p2p_sim[(m * criterion.max_proxy_per_class):((m + 1) * criterion.max_proxy_per_class),
                                (m * criterion.max_proxy_per_class):((m + 1) * criterion.max_proxy_per_class)].double().detach().cpu().numpy()
                        low_dim_proxy_p2psim.append(p2p_sim_m[:int(count_proxy[m]), :int(count_proxy[m])])

            low_dim_proxy_sub = np.asarray(low_dim_proxy_sub)

            visualize_interactive(images,
                                low_dim_emb_sub, low_dim_proxy_sub,
                                label_sub, subclasses, label_cmap,
                                base_loss_sub,
                                gt_D_weighted_sub,
                                gt_prob_sub,
                                low_dim_proxy_p2psim,
                                plot_dir, i,
                                interactive)


    '''Line plot'''
    # with open("{0}/{1}_ip.json".format('log', '{}_{}_trainval_{}_0_{}'.format(dataset_name, dataset_name, sz_embedding, dynamic_proxy)), 'rt') as handle:
    #     cache_sim = json.load(handle)
    # with open("{0}/{1}_cls.json".format('log', '{}_{}_trainval_{}_0_{}'.format(dataset_name, dataset_name, sz_embedding, dynamic_proxy)), 'rt') as handle:
    #     cache_label = json.load(handle)

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
