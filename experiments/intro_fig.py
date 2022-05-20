import matplotlib.pyplot as plt

from Influence_function.influence_function import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    loss_type = 'ProxyNCA_prob_orig'; sz_embedding = 512; epoch = 40; test_crop = False
    dataset_name = 'cub';  config_name = 'cub'; seed = 0
    # dataset_name = 'cars'; config_name = 'cars'; seed = 3
    # dataset_name = 'inshop'; config_name = 'inshop'; seed = 4

    '''============ Our Influence function =================='''
    IS = MCScalableIF(dataset_name, seed, loss_type, config_name, test_crop)
    nb_classes = IS.dl_ev.dataset.nb_classes()
    print(nb_classes)
    # confusion_class_pairs = IS.get_confusion_class_pairs()

    cub_nca = [[172, 143, 145, 128, 126, 124, 178, 116, 122, 130],
               [128, 143, 117, 195, 178, 122, 126, 116, 172, 120],
               [172, 143, 178, 126, 116, 144, 130, 122, 117, 196],
               [128, 172, 143, 178, 130, 117, 195, 111, 118, 116],
               [172, 143, 178, 145, 117, 116 ,128, 130, 196, 126]]
    cars_nca = [[103, 163, 183, 187, 186, 111 ,178, 137, 179 ,182],
                [163, 103, 182, 183, 137 ,179 ,148 ,139, 172 ,189],
                [183, 103, 166, 139, 179 ,163 ,128, 172, 176 ,186],
                [103, 163, 183, 179 ,187 ,139 ,166, 176, 186 ,177],
                [103, 183, 178, 182 ,102 ,147 ,186, 163, 187 ,135]]
    inshop_nca = [[6070, 5077, 4880, 7403, 7841, 7271, 7581, 4897, 5677, 4768],
                  [6070, 7403, 5077, 4880, 7841, 7581, 7271, 4897, 4768, 5677],
                  [6070, 5077, 4880, 7841, 7403, 4897, 7271, 7581, 4768, 5677],
                  [6070, 4880, 7403, 5077, 7841, 7581, 4897, 4768, 7271, 7619],
                  [6070, 4880, 5077, 7403, 7841, 4897, 7581, 7271, 4768, 5677]]

    heatmap = np.empty((5, nb_classes))
    heatmap[:] = np.nan
    for it, cars_nca_seed in enumerate(cub_nca):
        heatmap[it, np.asarray(cars_nca_seed)-nb_classes] = 1.
    # Plot the heatmap, customize and label the ticks
    fig = plt.figure(figsize=(30, 20))
    ax = fig.add_subplot(311)
    im = ax.imshow(heatmap, interpolation='nearest', cmap='Oranges_r', aspect='auto')
    ax.set_ylabel('ProxyNCA++', fontdict={'fontsize': 50})
    ax.set_yticks(range(5))
    ax.set_yticklabels(['Seed 0', 'Seed 1', 'Seed 2', 'Seed 3', 'Seed 4'], fontdict={'fontsize': 30})
    ax.set_xticks([])

    cub_anchor = [[116, 178, 145, 120, 172, 130, 117 ,126 ,143, 128],
               [143, 111, 178, 142, 117, 145 ,126 ,116, 134, 128],
               [196, 143, 178, 172, 145, 116 ,120 ,128, 122, 144],
               [143, 178, 116, 115, 172, 118 ,120 ,117, 144 ,126],
               [142, 117, 120, 145, 178 ,126, 172, 196 ,116, 195]]
    cars_anchor = [[103, 163 ,182 ,183 ,139, 111, 102, 176, 179, 166],
                   [103 ,163 ,166 ,176 ,182 ,139, 183, 124, 102, 172],
                   [103 ,183 ,163 ,179 ,182 ,166, 186, 102, 124, 176],
                   [103 ,186 ,176 ,182 ,150, 180, 124, 128, 187, 163],
                   [182 ,163 ,103 ,102, 186, 178, 176 ,183, 187 ,124]]
    inshop_anchor = [[6070, 4880, 5077, 7841, 4897, 7403, 7271, 7581, 6739 ,5677],
                     [6070, 5077, 4880, 7841, 7403, 4897, 7271, 7581, 4768 ,5677],
                     [5077, 6070, 4880, 7841, 7403, 4897, 7271 ,7581, 5677 ,4768],
                     [6070, 4880, 5077, 7841, 7403, 4897 ,7581, 7271, 5677 ,4768],
                     [5077, 6070, 4880, 7403, 7841, 7581 ,4897, 5677 ,4768 ,7271]]
    heatmap = np.empty((5, nb_classes))
    heatmap[:] = np.nan
    for it, cub_anchor_seed in enumerate(cub_anchor):
        heatmap[it, np.asarray(cub_anchor_seed) - nb_classes] = 1.
    # Plot the heatmap, customize and label the ticks
    ax = fig.add_subplot(312)
    im = ax.imshow(heatmap, interpolation='nearest', cmap='Blues_r', aspect='auto')
    ax.set_ylabel('ProxyAnchor', fontdict={'fontsize': 50})
    ax.set_yticks(range(5))
    ax.set_yticklabels(['Seed 0', 'Seed 1', 'Seed 2', 'Seed 3', 'Seed 4'], fontdict={'fontsize': 30})
    ax.set_xticks([])

    cub_softriple = [[172, 178, 126, 128, 143, 117, 156, 122, 196, 145],
               [172, 128, 178, 143, 126, 117 ,116 ,195 ,144, 120],
               [143, 128 ,178, 172, 118 ,126, 116, 117 ,144 ,195],
               [172, 178 ,143, 117, 116, 126, 195, 120 ,128 ,130],
               [117, 178, 143, 126, 172, 128, 118, 130 ,142 ,196]]
    cars_softtriple = [[163, 103, 179, 186, 183, 176, 187, 143, 102, 182],
                       [163 ,103, 183, 187, 139, 128, 176, 106, 175, 178],
                       [166, 163, 103, 147, 183, 111, 139, 176, 123, 186],
                       [103 ,139, 147, 163, 179, 166, 183, 186, 178, 187],
                       [183, 139, 187, 163, 103, 143, 182, 176, 147, 179]]
    inshop_softtriple = [[5077, 6070, 7841, 7403, 4880, 4897, 7271, 5677, 7581, 4768],
                         [5077, 7403, 6070, 4880, 7841, 4897, 7581, 4768, 7271, 5677],
                         [5077 ,6070, 7403, 4880, 7841, 4897, 7271, 7581, 4768 ,5677],
                         [5077 ,6070, 4880, 7403, 7841, 4897, 7271, 7581, 4768 ,5677],
                         [5077 ,6070, 7403, 4880, 7841, 4897, 7581, 5677, 4768 ,7271]]

    heatmap = np.empty((5, nb_classes))
    heatmap[:] = np.nan
    for it, cub_softtriple_seed in enumerate(cub_softriple):
        heatmap[it, np.asarray(cub_softtriple_seed) - nb_classes] = 1.
    # Plot the heatmap, customize and label the ticks
    ax = fig.add_subplot(313)
    im = ax.imshow(heatmap, interpolation='nearest', cmap='Greens_r', aspect='auto')
    ax.set_ylabel('SoftTriple', fontdict={'fontsize': 50})
    ax.set_yticks(range(5))
    ax.set_yticklabels(['Seed 0', 'Seed 1', 'Seed 2', 'Seed 3', 'Seed 4'], fontdict={'fontsize': 30})

    plt.xlabel("Testing Classes", fontdict={'fontsize': 50})
    plt.tight_layout()
    # plt.show()
    plt.savefig("images/generalization_error_{}.png".format(dataset_name))

