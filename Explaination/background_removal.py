import PIL
import matplotlib.pyplot as plt
from rembg.bg import remove
import numpy as np
import io
from PIL import Image
from PIL import ImageFile
from Influence_function.influential_sample import InfluentialSample
import torch
import os
import scipy

def remove_background(input_path):

    # Uncomment the following line if working with trucated image formats (ex. JPEG / JPG)
    # if input_path.endswith('jpg') or input_path.endswith('jpeg') or input_path.endswith('JPG') or input_path.endswith('JPEG'):
    #     ImageFile.LOAD_TRUNCATED_IMAGES = True
    # else:
    #     ImageFile.LOAD_TRUNCATED_IMAGES = False

    f = np.fromfile(input_path)
    result = remove(f)
    img = Image.open(io.BytesIO(result)).convert("RGBA")

    return img


if __name__ == '__main__':

    dataset_name = 'cub'
    loss_type = 'ProxyNCA_pfix'
    config_name = 'cub'
    sz_embedding = 512
    seed = 4
    measure = 'confusion'
    epoch = 40
    i = 143; j = 140
    #
    IS = InfluentialSample(dataset_name, seed, loss_type, config_name, measure, True, sz_embedding, epoch)

    indices_cls1 = torch.arange(len(IS.testing_embedding))[IS.testing_label == i]
    indices_cls2 = torch.arange(len(IS.testing_embedding))[IS.testing_label == j]

    # for path in [IS.dl_ev.dataset.im_paths[x] for x in indices_cls1]:
    #     img_after = remove_background(path)
    #     plt.imshow(img_after)
    #     plt.show()
    # TODO IS.dl_ev.dataset.im_paths[indices_cls2]

    '''CUB background removal'''
    # ct = 0
    # for folder in os.listdir('./mnt/datasets/CUB_200_2011/images (copy)'):
    #     for file in os.listdir(os.path.join('./mnt/datasets/CUB_200_2011/images (copy)', folder)):
    #         file_path = os.path.join('./mnt/datasets/CUB_200_2011/images (copy)', folder, file)
    #         try:
    #             check = Image.open(file_path)
    #         except PIL.UnidentifiedImageError:
    #             print(file_path, "unidentifiable image")
    #             continue
    #
    #         img_after = remove_background(file_path)
    #         print(file_path)
    #         plt.clf()
    #         plt.imshow(img_after); plt.axis('off'); plt.tight_layout()
    #         plt.savefig(file_path)
    #         plt.close()
    #         plt.clf()
    #         # plt.show()
    #         ct += 1
    #         # if ct >= 5:
    #         #     exit()

    '''CARS background removal'''
    # annos_fn = './mnt/datasets/CARS_196/cars_annos.mat'
    # cars = scipy.io.loadmat(annos_fn)
    # print(cars)

    ct = 0
    for file in os.listdir('./mnt/datasets/CARS_196/car_ims (copy)'):
        file_path = os.path.join('./mnt/datasets/CARS_196/car_ims (copy)', file)
        try:
            check = Image.open(file_path)
        except PIL.UnidentifiedImageError:
            print(file_path, "unidentifiable image")
            continue

        img_after = remove_background(file_path)
        print(file_path)
        plt.clf()
        plt.imshow(img_after); plt.axis('off'); plt.tight_layout()
        plt.savefig(file_path)
        # plt.show()
        plt.close()
        plt.clf()
        ct += 1
#         # if ct >= 5:
#         #     exit()


