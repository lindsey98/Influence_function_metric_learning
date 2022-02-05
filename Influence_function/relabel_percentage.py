import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image
from torchvision.io.image import read_image
from Influence_function.influence_function import MCScalableIF
from explaination.CAM_methods import *
from Influence_function.EIF_utils import *
from Influence_function.IF_utils import *
from utils import overlay_mask
from evaluation import assign_by_euclidian_at_k_indices
import sklearn
import pickle
from utils import evaluate
from Influence_function.sample_relabel import SampleRelabel, kNN_label_pred
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == '__main__':

    loss_type = 'ProxyNCA_prob_orig'; sz_embedding = 512; epoch = 40; test_crop = False
    # dataset_name = 'cub';  config_name = 'cub'; seed = 0
    # dataset_name = 'cars'; config_name = 'cars'; seed = 3
    dataset_name = 'inshop'; config_name = 'inshop'; seed = 4
    # dataset_name = 'sop'; config_name = 'sop'; seed = 3

    IS = SampleRelabel(dataset_name, seed, loss_type, config_name, test_crop)

    relabel_dict = []
    unique_labels, unique_counts = torch.unique(IS.train_label, return_counts=True)
    median_shots_percls = unique_counts.median().item()
    _, prob_relabel = kNN_label_pred(query_indices=np.arange(len(IS.dl_tr.dataset)), embeddings=IS.train_embedding,
                                     labels=IS.train_label,
                                     nb_classes=IS.dl_tr.dataset.nb_classes(), knn_k=median_shots_percls)

    for kk in range(len(IS.dl_tr.dataset)):
        relabel_dict.append(prob_relabel[kk].detach().cpu().numpy().argmax())
    relabel_dict = np.asarray(relabel_dict)

    # Distribution of each class being relabelled as
    relabel_dist = []
    for cls in range(IS.dl_tr.dataset.nb_classes()):
        cls_indices = np.where(np.asarray(IS.dl_tr.dataset.ys) == cls)[0]
        relabel_dist.append(relabel_dict[cls_indices])
    relabel_dist = np.asarray(relabel_dist)

    ct_relabel = 0
    ct_relabel_sig = 0
    centrailized_relabel = []
    diversified_relabel = []
    individual_relabel = []
    for cls in range(IS.dl_tr.dataset.nb_classes()):
        if np.sum(relabel_dist[cls] != cls) > 0:
            ct_relabel += 1
            percentage = np.sum(relabel_dist[cls] != cls) / len(relabel_dist[cls])
            if percentage >= 0.1:
                # print('Cls {} Percentage {}'.format(cls, percentage))
                ct_relabel_sig += 1
                relabel_freq = np.asarray([np.sum(relabel_dist[cls] == c) / len(relabel_dist[cls]) \
                                              for c in range(IS.dl_tr.dataset.nb_classes())])
                high_freq_relabel_cls = relabel_freq.argsort()[::-1]
                if relabel_freq[high_freq_relabel_cls[1]] >= 0.9 * percentage: # centralized
                    centrailized_relabel.append(cls)
                else:
                    diversified_relabel.append(cls)
            elif percentage <= 0.05 and percentage > 0:
                individual_relabel.append(cls)

    print('Total number of classes {}'.format(IS.dl_tr.dataset.nb_classes()))
    print('Total number of classes with relabelling suggestion {}'.format(ct_relabel))
    print('Total number of classes with significant (>=10%) relabelling suggestion {}'.format(ct_relabel_sig))
    print('Centralized', centrailized_relabel)
    print('Diversified', diversified_relabel)
    print('Individual', individual_relabel)

    '''Set an entropy threshold to distinguish centralized and diversified'''
    # central_entropy = []
    # for cls in centrailized_relabel:
    #     relabel_freq = np.asarray([np.sum(relabel_dist[cls] == c) / len(relabel_dist[cls]) for c in range(IS.dl_tr.dataset.nb_classes())])
    #     high_freq_relabel_cls = relabel_freq.argsort()[::-1]
    #     relabel_freq_ = relabel_freq[relabel_freq.argsort()[::-1][1:][relabel_freq[relabel_freq.argsort()[::-1][1:]] > 0]]
    #     relabel_entropy = -sum(relabel_freq_ * np.log(relabel_freq_))
    #     central_entropy.append(relabel_entropy)
    # print(central_entropy)
    #
    # entropy = []
    # for cls in diversified_relabel:
    #     relabel_freq = np.asarray([np.sum(relabel_dist[cls] == c) / len(relabel_dist[cls]) for c in range(IS.dl_tr.dataset.nb_classes())])
    #     high_freq_relabel_cls = relabel_freq.argsort()[::-1]
    #     relabel_freq_ = relabel_freq[relabel_freq.argsort()[::-1][1:][relabel_freq[relabel_freq.argsort()[::-1][1:]] > 0]]
    #     relabel_entropy = -sum(relabel_freq_ * np.log(relabel_freq_))
    #     entropy.append(relabel_entropy)
    # print(entropy)

    # CUB
    # Centralized [29, 53]
    # Diversified [58, 61, 96]
    # Individual [1, 2, 8, 10, 12, 14, 22, 28, 30, 31, 32, 37, 39, 48, 49, 50, 66, 67, 68, 77, 78, 79, 81, 82, 85, 88, 95, 97]

    # CARS
    # Centralized[12, 21, 45, 63, 68, 70, 82, 83]
    # Diversified[1, 13, 15, 18]
    # Individual[2, 4, 6, 7, 9, 10, 16, 17, 20, 28, 35, 43, 47, 53, 56, 61, 65, 69, 72, 74, 79, 81, 85, 86, 89, 94]

    # InShop
    # Centralized [23, 24, 25, 28, 42, 44, 49, 53, 61, 75, 78, 79, 82, 85, 89, 93, 99, 101, 105, 107, 109, 114, 131, 134, 135, 140, 143, 147, 151, 153, 156, 158, 171, 172, 175, 184, 195, 199, 211, 220, 230, 238, 242, 245, 248, 257, 260, 269, 272, 274, 277, 281, 290, 292, 296, 298, 306, 313, 323, 327, 336, 337, 345, 369, 377, 378, 385, 395, 397, 410, 412, 416, 423, 426, 430, 432, 443, 444, 447, 450, 452, 458, 461, 477, 480, 482, 484, 487, 494, 496, 500, 505, 506, 518, 543, 544, 545, 546, 562, 569, 572, 581, 604, 606, 615, 619, 641, 657, 662, 664, 676, 677, 697, 720, 723, 740, 743, 744, 745, 749, 753, 755, 761, 774, 789, 811, 830, 833, 841, 845, 858, 866, 867, 869, 871, 874, 876, 877, 879, 882, 894, 896, 925, 930, 931, 932, 936, 940, 945, 946, 950, 953, 956, 962, 968, 979, 981, 982, 987, 1008, 1012, 1020, 1040, 1042, 1049, 1051, 1056, 1062, 1067, 1079, 1081, 1082, 1096, 1114, 1116, 1117, 1120, 1125, 1134, 1135, 1143, 1157, 1166, 1169, 1181, 1200, 1202, 1207, 1210, 1217, 1220, 1221, 1225, 1226, 1234, 1255, 1257, 1260, 1266, 1268, 1273, 1277, 1280, 1284, 1285, 1294, 1304, 1308, 1309, 1312, 1318, 1325, 1336, 1338, 1339, 1340, 1349, 1350, 1363, 1364, 1382, 1405, 1408, 1409, 1418, 1433, 1444, 1445, 1450, 1454, 1458, 1462, 1467, 1480, 1483, 1485, 1492, 1502, 1506, 1511, 1513, 1519, 1524, 1526, 1536, 1537, 1541, 1550, 1559, 1562, 1569, 1574, 1583, 1591, 1593, 1598, 1616, 1621, 1626, 1640, 1644, 1646, 1653, 1654, 1658, 1670, 1674, 1688, 1689, 1695, 1696, 1716, 1717, 1719, 1734, 1744, 1749, 1750, 1751, 1765, 1767, 1772, 1778, 1785, 1791, 1792, 1823, 1829, 1871, 1880, 1882, 1888, 1903, 1905, 1908, 1915, 1916, 1921, 1928, 1929, 1940, 1949, 1974, 1983, 1984, 2003, 2015, 2018, 2024, 2025, 2027, 2031, 2032, 2035, 2037, 2039, 2045, 2055, 2059, 2083, 2085, 2086, 2090, 2105, 2112, 2113, 2116, 2120, 2128, 2134, 2137, 2142, 2156, 2157, 2165, 2184, 2192, 2203, 2209, 2222, 2225, 2227, 2233, 2236, 2239, 2245, 2250, 2251, 2263, 2279, 2285, 2297, 2306, 2308, 2310, 2328, 2332, 2334, 2335, 2342, 2348, 2354, 2368, 2398, 2405, 2423, 2431, 2458, 2463, 2466, 2468, 2479, 2484, 2489, 2504, 2510, 2512, 2523, 2533, 2535, 2536, 2544, 2550, 2554, 2560, 2565, 2572, 2576, 2590, 2591, 2592, 2598, 2600, 2602, 2603, 2634, 2636, 2639, 2644, 2646, 2654, 2656, 2659, 2673, 2685, 2695, 2699, 2700, 2706, 2715, 2725, 2727, 2730, 2732, 2739, 2750, 2754, 2757, 2761, 2781, 2783, 2791, 2792, 2806, 2811, 2815, 2816, 2820, 2822, 2825, 2829, 2832, 2837, 2841, 2846, 2855, 2858, 2861, 2868, 2876, 2879, 2884, 2896, 2905, 2906, 2910, 2911, 2915, 2921, 2939, 2953, 2974, 2979, 2991, 2993, 3003, 3018, 3021, 3025, 3034, 3038, 3043, 3051, 3062, 3069, 3076, 3083, 3090, 3095, 3106, 3109, 3118, 3119, 3123, 3132, 3151, 3159, 3162, 3163, 3177, 3179, 3182, 3191, 3196, 3197, 3198, 3211, 3219, 3222, 3224, 3236, 3237, 3238, 3247, 3260, 3264, 3267, 3280, 3285, 3288, 3295, 3299, 3303, 3304, 3314, 3320, 3345, 3348, 3351, 3354, 3359, 3365, 3366, 3373, 3374, 3385, 3388, 3417, 3420, 3423, 3429, 3430, 3456, 3459, 3461, 3464, 3467, 3481, 3486, 3487, 3492, 3496, 3502, 3503, 3508, 3510, 3513, 3519, 3541, 3551, 3562, 3564, 3582, 3589, 3590, 3600, 3605, 3622, 3635, 3645, 3651, 3652, 3659, 3660, 3665, 3667, 3674, 3691, 3694, 3699, 3705, 3706, 3708, 3716, 3727, 3735, 3743, 3746, 3748, 3749, 3750, 3752, 3766, 3771, 3790, 3791, 3793, 3798, 3799, 3807, 3808, 3813, 3818, 3819, 3841, 3851, 3860, 3867, 3876, 3890, 3899, 3903, 3915, 3916, 3927, 3941, 3949, 3951, 3968, 3969, 3987, 3990]
    # Diversified [103, 194, 247, 284, 285, 288, 325, 367, 388, 428, 437, 445, 448, 459, 491, 495, 528, 531, 532, 573, 601, 621, 622, 638, 640, 647, 655, 658, 660, 693, 709, 721, 728, 729, 734, 764, 768, 776, 851, 878, 903, 905, 909, 915, 927, 949, 952, 957, 971, 989, 1000, 1001, 1036, 1060, 1066, 1098, 1100, 1152, 1155, 1158, 1165, 1191, 1228, 1237, 1247, 1250, 1252, 1296, 1315, 1358, 1387, 1390, 1410, 1426, 1443, 1507, 1508, 1534, 1601, 1611, 1613, 1634, 1677, 1703, 1713, 1758, 1761, 1780, 1796, 1801, 1803, 1815, 1840, 1858, 1864, 1865, 1870, 1885, 1934, 1950, 1978, 1991, 1999, 2000, 2002, 2026, 2049, 2076, 2100, 2115, 2132, 2133, 2144, 2181, 2255, 2286, 2296, 2309, 2343, 2365, 2401, 2421, 2444, 2473, 2480, 2486, 2491, 2497, 2501, 2502, 2586, 2589, 2605, 2609, 2619, 2645, 2671, 2681, 2714, 2789, 2795, 2802, 2835, 2845, 2850, 2888, 2893, 2912, 2931, 2935, 2938, 2946, 2951, 2961, 3019, 3063, 3066, 3075, 3078, 3084, 3092, 3208, 3220, 3235, 3287, 3313, 3341, 3376, 3397, 3399, 3411, 3419, 3425, 3441, 3453, 3474, 3494, 3500, 3509, 3518, 3533, 3572, 3577, 3640, 3646, 3666, 3675, 3678, 3712, 3772, 3812, 3828, 3856, 3883, 3889, 3891, 3908, 3971, 3973, 3975, 3995]
    # Individual [48, 611, 682, 967, 1497, 1629, 1723, 1730, 1733, 1753, 1962, 2369, 2462, 2529, 2616, 3102, 3125, 3803, 3965]

    '''Plot'''
    cls = 682
    cls_indices = np.where(np.asarray(IS.dl_tr.dataset.ys) == cls)[0]
    relabel_as_classes = relabel_dict[cls_indices]

    relabel_freq = np.asarray([np.sum(relabel_dist[cls] == c) / len(relabel_dist[cls]) for c in range(IS.dl_tr.dataset.nb_classes())])
    high_freq_relabel_cls = relabel_freq.argsort()[::-1]

    for topk in range(1, IS.dl_tr.dataset.nb_classes()):
        if relabel_freq[high_freq_relabel_cls[topk]] == 0:
            break
        # top-1 class
        topk_relabel_class = np.arange(IS.dl_tr.dataset.nb_classes())[high_freq_relabel_cls[topk]]
        interest_indices = np.where(relabel_as_classes == topk_relabel_class)[0]
        vis_indices = cls_indices[interest_indices]
        fig = plt.figure(figsize=(30, 15))
        for kk in range(min(len(vis_indices), 10)):
            plt.subplot(2, 10, kk+1)
            img = read_image(IS.dl_tr.dataset.im_paths[vis_indices[kk]])
            plt.imshow(to_pil_image(img))
            plt.title("Ind = {} Cls = {}".format(vis_indices[kk], cls))
            plt.axis('off')
            # plt.tight_layout()

        ind_cls = np.where(np.asarray(IS.dl_tr.dataset.ys) == topk_relabel_class)[0]
        for i in range(min(5, len(ind_cls))):
            plt.subplot(2, 10, i + 11)
            img = read_image(IS.dl_tr.dataset.im_paths[ind_cls[i]])
            plt.imshow(to_pil_image(img))
            plt.title('Ind = {} Cls = {}'.format(ind_cls[i], topk_relabel_class))
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close()



    # for cls in range(IS.dl_tr.dataset.nb_classes()):
    #     if len(np.unique(relabel_dist[cls])) == 5:
    #         fig = plt.figure(figsize=(30, 15))
    #         for kk, unique_cls in enumerate(np.unique(relabel_dist[cls])):
    #             plt.subplot(1, 5, kk+1)
    #             ind = np.where(np.asarray(IS.dl_tr.dataset.ys) == unique_cls)[0][0]
    #             img = read_image(IS.dl_tr.dataset.im_paths[ind])
    #             plt.imshow(to_pil_image(img))
    #             # plt.title("Ind = {}".format(ind))
    #             plt.axis('off')
    #             plt.tight_layout()
    #         plt.savefig("Inshop_{}_{}_{}_{}_{}.png".format(np.unique(relabel_dist[cls])[0],
    #                                                        np.unique(relabel_dist[cls])[1],
    #                                                        np.unique(relabel_dist[cls])[2],
    #                                                        np.unique(relabel_dist[cls])[3],
    #                                                        np.unique(relabel_dist[cls])[4]),bbox_inches='tight')
    #
    #


