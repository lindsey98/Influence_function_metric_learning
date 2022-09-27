import matplotlib.pyplot as plt
import os
from warnings import warn, catch_warnings, filterwarnings

try:
    import tensorflow as tf
except ImportError:
    warn(
        """The umap.parametric_umap package requires Tensorflow > 2.0 to be installed.
    You can install Tensorflow at https://www.tensorflow.org/install

    or you can install the CPU version of Tensorflow using 
    pip install umap-learn[parametric_umap]
    """
    )
    raise ImportError("umap.parametric_umap requires Tensorflow >= 2.0") from None

TF_MAJOR_VERSION = int(tf.__version__.split(".")[0])
if TF_MAJOR_VERSION < 2:
    warn(
        """The umap.parametric_umap package requires Tensorflow > 2.0 to be installed.
    You can install Tensorflow at https://www.tensorflow.org/install

    or you can install the CPU version of Tensorflow using 
    pip install umap-learn[parametric_umap]
    """
    )
    raise ImportError("umap.parametric_umap requires Tensorflow >= 2.0") from None

from umap.parametric_umap import ParametricUMAP
from loss import *
from utils import predict_batchwise
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, \
    AnnotationBbox
import evaluation
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from Influence_function.influence_function import EIF
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


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
        torch.save(embedding, '{}/Epoch_{}/training_embeddings.pth'.format(model_dir, e))
        torch.save(label, '{}/Epoch_{}/training_labels.pth'.format(model_dir, e))
        torch.save(indices, '{}/Epoch_{}/training_indices.pth'.format(model_dir, e))

        testing_embedding, testing_label, testing_indices = predict_batchwise(model, dl_ev)
        torch.save(testing_embedding, '{}/Epoch_{}/testing_embeddings.pth'.format(model_dir, e))
        torch.save(testing_label, '{}/Epoch_{}/testing_labels.pth'.format(model_dir, e))
        torch.save(testing_indices, '{}/Epoch_{}/testing_indices.pth'.format(model_dir, e))

    else:
        embedding = torch.load('{}/Epoch_{}/training_embeddings.pth'.format(model_dir, e))
        label = torch.load('{}/Epoch_{}/training_labels.pth'.format(model_dir, e))

        testing_embedding = torch.load('{}/Epoch_{}/testing_embeddings.pth'.format(model_dir, e))
        testing_label = torch.load('{}/Epoch_{}/testing_labels.pth'.format(model_dir, e))

    # need to normalize, otherwise producing wierd results
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
                              verbose=False,
                              batch_size=64)

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
                                               # stacked_proxies.cpu().numpy(),
                                               testing_embedding.cpu().numpy()), axis=0))
        embedder.encoder.save('{}/Epoch_{}/parametric_model/encoder'.format(model_dir, e))
    else:
        embedder.encoder = tf.keras.models.load_model('{}/Epoch_{}/parametric_model/encoder'.format(model_dir, e))

    # transform high dimensional embedding and proxy to low-dimension
    low_dim_emb = embedder.transform(embedding.detach().cpu().numpy())
    low_dim_emb_test = embedder.transform(testing_embedding.detach().cpu().numpy())
    low_dim_proxy = embedder.transform(stacked_proxies.detach().cpu().numpy())

    return embedder, \
           low_dim_proxy,\
           (low_dim_emb, label),\
           (low_dim_emb_test, testing_label)


if __name__ == '__main__':

    noisy_level = 0.1; sz_embedding = 512; epoch = 40; test_crop = False
    loss_type = 'ProxyNCA_prob_orig_noisy_{}'.format(noisy_level); dataset_name = 'cub_noisy'; config_name = 'cub_ProxyNCA_prob_orig'; seed = 0
    IS = EIF(dataset_name, seed, loss_type, config_name, 'dataset/config.json', test_crop, sz_embedding, epoch, 'ResNet', noisy_level)

    presaved = True
    pretrained = True
    interactive = True
    highlight = True

    folder = IS.folder
    plot_dir = '{}/resnet_{}_umap_plots'.format(folder, sz_embedding)

    os.makedirs(os.path.join(folder, 'Training_data'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'Testing_data'), exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    embedder, low_dim_proxy, (low_dim_emb_train, train_label), (low_dim_emb_test, test_label) = pumap_training(model=IS.model,
                                                                                                               model_dir=IS.model_dir,
                                                                                                               e=epoch,
                                                                                                               stacked_proxies=IS.criterion.proxies,
                                                                                                               dl_tr=IS.dl_tr,
                                                                                                               dl_ev=IS.dl_ev,
                                                                                                               presaved=presaved,
                                                                                                               pretrained=pretrained)
    '''Visualize'''
    # belong to top1 wrong classes
    topk_wrong_classes = [IS.get_confusion_class_pairs(topk_cls=1)[0][0][0]]
    topk_wrong_class_ind = np.where(np.isin(np.asarray(test_label), topk_wrong_classes))[0]
    topk_label_sub_test = test_label[topk_wrong_class_ind].numpy()
    topk_low_dim_emb_test = low_dim_emb_test[topk_wrong_class_ind, :]
    images_test = [to_pil_image(read_image(IS.dl_ev.dataset.im_paths[ind])) for ind in topk_wrong_class_ind]

    # helpful noisy data
    influence_values = np.load("MislabelExp_Influential_data/{}_{}_influence_values_testcls{}_SIF_theta{}_{}_step50.npy".format(IS.dataset_name, IS.loss_type, 0, 1, noisy_level))
    training_sample_by_influence = influence_values.argsort()[::-1]
    gt_mislabelled_indices = IS.dl_tr.dataset.noisy_indices
    overlap = np.isin(training_sample_by_influence, gt_mislabelled_indices)
    confusion_class_pairs = IS.get_confusion_class_pairs()
    training_sample_by_influence_in_mislabelled = [training_sample_by_influence[ct] for ct in range(len(training_sample_by_influence)) \
                                                   if overlap[ct]==True]

    # helpful_noisy_indices = np.asarray(training_sample_by_influence_in_mislabelled[::-1][:10])
    helpful_noisy_indices = np.asarray(training_sample_by_influence_in_mislabelled[:10])
    helpful_noisy_orig_class = np.asarray(IS.dl_tr_clean.dataset.ys)[helpful_noisy_indices]
    helpful_noisy_relabel_class = np.asarray(IS.dl_tr.dataset.ys)[helpful_noisy_indices]
    low_dim_emb_helpful_noisy_train = low_dim_emb_train[helpful_noisy_indices, :]
    images_helpful_noisy_train = [to_pil_image(read_image(IS.dl_tr.dataset.im_paths[ind])) for ind in helpful_noisy_indices]

    # clean data falling into relabelled classes
    clean_train_indices = np.asarray(np.where(np.isin(np.asarray(IS.dl_tr_clean.dataset.ys), helpful_noisy_relabel_class))[0].tolist() + \
                                     np.where(np.isin(np.asarray(IS.dl_tr_clean.dataset.ys), helpful_noisy_orig_class))[0].tolist())
    clean_train_class = np.asarray(IS.dl_tr_clean.dataset.ys)[clean_train_indices]
    low_dim_emb_clean_train = low_dim_emb_train[clean_train_indices, :]
    images_train = [to_pil_image(read_image(IS.dl_tr_clean.dataset.im_paths[ind])) for ind in clean_train_indices]

    # Visualize
    fig, ax = plt.subplots()
    x_test, y_test = topk_low_dim_emb_test[:, 0], topk_low_dim_emb_test[:, 1]
    x_helpful_noisy_train, y_helpful_noisy_train = low_dim_emb_helpful_noisy_train[:, 0], low_dim_emb_helpful_noisy_train[:, 1]
    x_clean_train, y_clean_train = low_dim_emb_clean_train[:, 0], low_dim_emb_clean_train[:, 1]

    line4ev = ax.scatter(x_test, y_test, c='orange', s=5)
    line4clean_tr = ax.scatter(x_clean_train, y_clean_train, c='blue', s=5)
    line4helpful_noisy_tr = ax.scatter(x_helpful_noisy_train, y_helpful_noisy_train, c='red', s=7)

    xybox = (-50., -50.)
    xybox_ac = (50., 50.)
    imagebox2 = OffsetImage(to_pil_image(read_image(IS.dl_ev.dataset.im_paths[0])), zoom=.3)
    ab = AnnotationBbox(imagebox2, (0, 0),
                        xybox=xybox,
                        xycoords='data',
                        boxcoords="offset points",
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)
    ab.set_visible(False)
    ac = ax.annotate("", xy=(0, 0),
                     xytext=xybox_ac, textcoords="offset points",
                     bbox=dict(boxstyle='round4', fc='linen', ec='k', lw=1),
                     arrowprops=dict(arrowstyle='->'))
    ax.add_artist(ac)
    ac.set_visible(False)

    sample_plots2 = []
    plot2 = ax.plot([], [], '.', ms=5, color='red', zorder=5)
    sample_plots2.append(plot2[0])
    plot2[0].set_visible(False)

    imagebox_helpful_noisy = OffsetImage(to_pil_image(read_image(IS.dl_tr.dataset.im_paths[0])), zoom=.3)
    ab_helpful_noisy = AnnotationBbox(imagebox_helpful_noisy, (0, 0),
                        xybox=xybox,
                        xycoords='data',
                        boxcoords="offset points",
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab_helpful_noisy)
    ab_helpful_noisy.set_visible(False)
    ac_helpful_noisy = ax.annotate("", xy=(0, 0),
                     xytext=xybox_ac, textcoords="offset points",
                     bbox=dict(boxstyle='round4', fc='linen', ec='k', lw=1),
                     arrowprops=dict(arrowstyle='->'))
    ax.add_artist(ac_helpful_noisy)
    ac_helpful_noisy.set_visible(False)


    imagebox_clean = OffsetImage(to_pil_image(read_image(IS.dl_tr.dataset.im_paths[0])), zoom=.3)
    ab_clean = AnnotationBbox(imagebox_clean, (0, 0),
                        xybox=xybox,
                        xycoords='data',
                        boxcoords="offset points",
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab_clean)
    ab_clean.set_visible(False)
    ac_clean = ax.annotate("", xy=(0, 0),
                     xytext=xybox_ac, textcoords="offset points",
                     bbox=dict(boxstyle='round4', fc='linen', ec='k', lw=1),
                     arrowprops=dict(arrowstyle='->'))
    ax.add_artist(ac_clean)
    ac_clean.set_visible(False)

    # sample_dot = []
    # dot = ax.plot([], [], '.', ms=5, color='blue', markeredgecolor='black', zorder=6)
    # sample_dot.append(dot[0])
    # dot[0].set_visible(False)

    def hover(event):
        # Helpful noisy Training
        if line4helpful_noisy_tr.contains(event)[0]:
            # find out the index within the array from the event
            ind = line4helpful_noisy_tr.contains(event)[1]["ind"][0]
            # get the figure size
            w, h = fig.get_size_inches() * fig.dpi
            ws = (event.x > w / 2.) * -1 + (event.x <= w / 2.)
            hs = (event.y > h / 2.) * -1 + (event.y <= h / 2.)
            # if event occurs in the top or right quadrant of the figure,
            # change the annotation box position relative to mouse.
            ab_helpful_noisy.xybox = (xybox[0] * ws, xybox[1] * hs)
            ab_helpful_noisy.set_visible(True)
            # place it at the position of the hovered scatter point
            ab_helpful_noisy.xy = (x_helpful_noisy_train[ind], y_helpful_noisy_train[ind])
            # set the image corresponding to that point
            imagebox_helpful_noisy.set_data(images_helpful_noisy_train[ind])
            ac_helpful_noisy.xybox = (xybox_ac[0] * -ws, xybox_ac[1] * -hs)
            ac_helpful_noisy.xy = (x_helpful_noisy_train[ind], y_helpful_noisy_train[ind])
            text = "Orig Label={} \n Relabelled as={} \n".format(helpful_noisy_orig_class[ind],
                                                                                    helpful_noisy_relabel_class[ind])
            ac_helpful_noisy.set_visible(True)
            ac_helpful_noisy.set_text(text)
        else:
            # if the mouse is not over a scatter point
            ab_helpful_noisy.set_visible(False)
            ac_helpful_noisy.set_visible(False)
        fig.canvas.draw_idle()

        # Clean Training
        if line4clean_tr.contains(event)[0]:
            # find out the index within the array from the event
            ind = line4clean_tr.contains(event)[1]["ind"][0]
            # get the figure size
            w, h = fig.get_size_inches() * fig.dpi
            ws = (event.x > w / 2.) * -1 + (event.x <= w / 2.)
            hs = (event.y > h / 2.) * -1 + (event.y <= h / 2.)
            # if event occurs in the top or right quadrant of the figure,
            # change the annotation box position relative to mouse.
            ab_clean.xybox = (xybox[0] * ws, xybox[1] * hs)
            ab_clean.set_visible(True)
            # place it at the position of the hovered scatter point
            ab_clean.xy = (x_clean_train[ind], y_clean_train[ind])
            # set the image corresponding to that point
            imagebox_clean.set_data(images_train[ind])
            ac_clean.xybox = (xybox_ac[0] * -ws, xybox_ac[1] * -hs)
            ac_clean.xy = (x_clean_train[ind], y_clean_train[ind])
            text = "Class Label={} \n".format(clean_train_class[ind])
            ac_clean.set_visible(True)
            ac_clean.set_text(text)
        else:
            # if the mouse is not over a scatter point
            ab_clean.set_visible(False)
            ac_clean.set_visible(False)
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
            ab.xybox = (xybox[0] * ws, xybox[1] * hs)
            ab.set_visible(True)
            # place it at the position of the hovered scatter point
            ab.xy = (x_test[ind], y_test[ind])
            # set the image corresponding to that point
            imagebox2.set_data(images_test[ind])
            if highlight:
                c = topk_label_sub_test[ind]
                data = topk_low_dim_emb_test[topk_label_sub_test == c, :]
                plot2[0].set_visible(True)
                sample_plots2[0].set_data(data.transpose())

            ac.xybox = (xybox_ac[0] * -ws, xybox_ac[1] * -hs)
            ac.xy = (x_test[ind], y_test[ind])
            text = "Testing data indices={} \n Label={} \n".format(topk_wrong_class_ind[ind],
                                                                   topk_label_sub_test[ind])
            ac.set_visible(True)
            ac.set_text(text)

        else:
            # if the mouse is not over a scatter point
            ab.set_visible(False)
            ac.set_visible(False)
            plot2[0].set_visible(False)
        fig.canvas.draw_idle()

    # # add callback for mouse moves
    fig.canvas.mpl_connect('motion_notify_event', hover)
    plt.draw()
    plt.show()

