
"""NOTE: I've checked the images for correctnes manually, i.e. each image
of gallery should have a corresponding pair in query (i.e. should look the
same) and also have the same label."""

#import PIL
#import torch
#import os
from .base import *

class InShop(BaseDatasetMod):
    """
    For the In-Shop Clothes Retrieval dataset, we use the predefined
    25, 882 training images of 3,997 classes for training. The test
    set is partitioned into a query set (14,218 images of 3,985 classes)
    and a gallery set (12, 612 images of 3, 985 classes)
    """
    def __init__(self, root, source, classes, transform, dset_type='train'):
        BaseDatasetMod.__init__(self, root, source, classes, transform)

        with open(
            os.path.join(
                root, 'Eval/list_eval_partition.txt'
            ), 'r'
        ) as f:
            lines = f.readlines()

        self.transform = transform

        # store for using later '__getitem__'
        self.dset_type = dset_type

        nb_samples = int(lines[0].strip('\n'))
        #print(nb_samples)
        assert nb_samples == 52712

        torch.utils.data.Dataset.__init__(self)
        self.or_im_paths = {'train': [], 'query': [], 'gallery': []}
        self.or_ys = {'train': [], 'query': [], 'gallery': []}
        or_I_ = {'train': 0, 'query': 0, 'gallery': 0}
        self.or_I = {'train': [], 'query': [], 'gallery': []}
        self.class_names_dict = {'WOMEN/Blouses_Shirts': 0, 'WOMEN/Cardigans': 1, 'WOMEN/Denim': 2, 'WOMEN/Dresses': 3, 'WOMEN/Graphic_Tees': 4,
                                 'WOMEN/Jackets_Coats': 5, 'WOMEN/Leggings': 6, 'WOMEN/Pants': 7, 'WOMEN/Rompers_Jumpsuits': 8,
                                 'WOMEN/Shorts': 9, 'WOMEN/Skirts': 10, 'WOMEN/Sweaters': 11, 'WOMEN/Sweatshirts_Hoodies': 12,
                                 'WOMEN/Tees_Tanks': 13,
                                 'MEN/Denim': 14, 'MEN/Jackets_Vests': 15, 'MEN/Pants': 16, 'MEN/Shirts_Polos': 17,
                                 'MEN/Shorts': 18,'MEN/Suiting': 19,'MEN/Sweaters': 20,'MEN/Sweatshirts_Hoodies': 21,
                                 'MEN/Tees_Tanks': 22}
        self.or_super_class_names = {'train': [], 'query': [], 'gallery': []}
        self.or_super_ys = {'train': [], 'query': [], 'gallery': []}


        print(dset_type)
        print(classes)

        # start from second line, since 0th and 1st contain meta-data
        for line in lines[2:]:
            im_path, im_id, eval_type = [
                l for l in line.split(' ') if l != '' and l != '\n']
            y = int(im_id.split('_')[1])
            # this is the old code chunk
            self.or_super_class_names[eval_type] += ['/'.join(im_path.split('/')[1:3])]
            self.or_super_ys[eval_type] += [self.class_names_dict['/'.join(im_path.split('/')[1:3])]]
            self.or_im_paths[eval_type] += [os.path.join(root, im_path)]
            self.or_ys[eval_type] += [y]
            self.or_I[eval_type] += [or_I_[eval_type]]
            or_I_[eval_type] += 1

        nb_samples_counted = len(self.or_im_paths['train']) + len(self.or_im_paths['gallery']) + len(self.or_im_paths['query'])
        assert nb_samples_counted == nb_samples

        # verify that labels are sorted for next step
        # self.or_ys['query'] == sorted(self.or_ys['query'])
        # self.or_ys['gallery'] == sorted(self.or_ys['gallery'])

        assert len(self.or_ys['train']) == 25882
        assert len(self.or_ys['query']) == 14218
        assert len(self.or_ys['gallery']) == 12612

        # verify that query and gallery have same labels
        assert set(self.or_ys['query']) == set(self.or_ys['gallery'])

        # labels of query and gallery are like [1, 1, 7, 7, 8, 11, ...]
        # condense them such that ordered without spaces,
        # i.e. 1 -> 1, 7 -> 2, ...
        # same thing for train labels
        idx_to_class = {idx: i for i, idx in enumerate(
            sorted(set(self.or_ys['train']))
        )}
        self.or_ys['train'] = list(
            map(lambda x: idx_to_class[x], self.or_ys['train']))

        # TODO: I change query labels to be incremental from training labels
        idx_to_class = {idx: i + len(set(self.or_ys['train'])) for i, idx in enumerate(
            sorted(set(self.or_ys['query']))
        )}
        for _type in ['query', 'gallery']:
            self.or_ys[_type] = list(
                map(lambda x: idx_to_class[x], self.or_ys[_type]))

        # should be 3997 classes for training, 3985 for query/gallery
        assert len(set(self.or_ys['train'])) == 3997
        assert len(set(self.or_ys['query'])) == 3985
        assert len(set(self.or_ys['gallery'])) == 3985

        ##start your dataset selection here for train, val and trainval 
        self.im_paths = []
        self.ys = []
        self.I = []
        self.super2ys = {}

        if dset_type == 'train':
            for ix in range(len(self.or_ys[dset_type])):
                y = self.or_ys[dset_type][ix]
                im_path = self.or_im_paths[dset_type][ix]
                ii = self.or_I[dset_type][ix]
                s_y = self.or_super_ys[dset_type][ix]
                if y not in self.super2ys.keys():
                    self.super2ys[y] = s_y
                if y in classes:
                    self.im_paths.append(im_path)
                    self.ys.append(y)
                    self.I.append(ii)

            for ix in range(len(self.or_ys['query'])): # add additional classes into training
                y = self.or_ys['query'][ix]
                im_path = self.or_im_paths['query'][ix]
                ii = self.or_I['query'][ix]
                s_y = self.or_super_ys['query'][ix]
                if y not in self.super2ys.keys():
                    self.super2ys[y] = s_y
                if y in classes:
                    self.im_paths.append(im_path)
                    self.ys.append(y)
                    self.I.append(ii)


            for ix in range(len(self.or_ys['gallery'])):
                y = self.or_ys['gallery'][ix]
                im_path = self.or_im_paths['gallery'][ix]
                ii = self.or_I['gallery'][ix]
                s_y = self.or_super_ys['gallery'][ix]
                if y not in self.super2ys.keys():
                    self.super2ys[y] = s_y
                if y in classes:
                    self.im_paths.append(im_path)
                    self.ys.append(y)
                    self.I.append(ii)
                    self.super_ys.append(s_y)

        elif dset_type in ['query', 'gallery']:
            for ix in range(len(self.or_ys[dset_type])):
                y = self.or_ys[dset_type][ix]
                im_path = self.or_im_paths[dset_type][ix]
                ii = self.or_I[dset_type][ix]
                s_y = self.or_super_ys[dset_type][ix]
                if y not in self.super2ys.keys():
                    self.super2ys[y] = s_y
                if y in classes:
                    self.im_paths.append(im_path)
                    self.ys.append(y)
                    self.I.append(ii)

        elif dset_type == 'all':
            for ix in range(len(self.or_ys['query'])):
                y = self.or_ys['query'][ix]
                im_path = self.or_im_paths['query'][ix]
                ii = self.or_I['query'][ix]
                s_y = self.or_super_ys['query'][ix]
                if y not in self.super2ys.keys():
                    self.super2ys[y] = s_y
                if y in classes:
                    self.im_paths.append(im_path)
                    self.ys.append(y)
                    self.I.append(ii)

            for ix in range(len(self.or_ys['gallery'])):
                y = self.or_ys['gallery'][ix]
                im_path = self.or_im_paths['gallery'][ix]
                ii = self.or_I['gallery'][ix]
                s_y = self.or_super_ys['gallery'][ix]
                if y not in self.super2ys.keys():
                    self.super2ys[y] = s_y
                if y in classes:
                    self.im_paths.append(im_path)
                    self.ys.append(y)
                    self.I.append(ii)

class InShopNoisy(BaseDatasetModNoisy):
    """
    For the In-Shop Clothes Retrieval dataset, we use the predefined
    25, 882 training images of 3,997 classes for training. The test
    set is partitioned into a query set (14,218 images of 3,985 classes)
    and a gallery set (12, 612 images of 3, 985 classes)
    """
    def __init__(self, root, source, classes, transform, dset_type='train', seed=0, mislabel_percentage=0.01):
        BaseDatasetModNoisy.__init__(self, root, source, classes, transform)

        with open(
            os.path.join(
                root, 'Eval/list_eval_partition.txt'
            ), 'r'
        ) as f:
            lines = f.readlines()

        self.transform = transform

        # store for using later '__getitem__'
        self.dset_type = dset_type

        nb_samples = int(lines[0].strip('\n'))
        #print(nb_samples)
        assert nb_samples == 52712

        torch.utils.data.Dataset.__init__(self)
        self.or_im_paths = {'train': [], 'query': [], 'gallery': []}
        self.or_ys = {'train': [], 'query': [], 'gallery': []}
        or_I_ = {'train': 0, 'query': 0, 'gallery': 0}
        self.or_I = {'train': [], 'query': [], 'gallery': []}
        self.class_names_dict = {'WOMEN/Blouses_Shirts': 0, 'WOMEN/Cardigans': 1, 'WOMEN/Denim': 2, 'WOMEN/Dresses': 3, 'WOMEN/Graphic_Tees': 4,
                                 'WOMEN/Jackets_Coats': 5, 'WOMEN/Leggings': 6, 'WOMEN/Pants': 7, 'WOMEN/Rompers_Jumpsuits': 8,
                                 'WOMEN/Shorts': 9, 'WOMEN/Skirts': 10, 'WOMEN/Sweaters': 11, 'WOMEN/Sweatshirts_Hoodies': 12,
                                 'WOMEN/Tees_Tanks': 13,
                                 'MEN/Denim': 14, 'MEN/Jackets_Vests': 15, 'MEN/Pants': 16, 'MEN/Shirts_Polos': 17,
                                 'MEN/Shorts': 18,'MEN/Suiting': 19,'MEN/Sweaters': 20,'MEN/Sweatshirts_Hoodies': 21,
                                 'MEN/Tees_Tanks': 22}
        self.or_super_class_names = {'train': [], 'query': [], 'gallery': []}
        self.or_super_ys = {'train': [], 'query': [], 'gallery': []}


        print(dset_type)
        print(classes)

        # start from second line, since 0th and 1st contain meta-data
        for line in lines[2:]:
            im_path, im_id, eval_type = [
                l for l in line.split(' ') if l != '' and l != '\n']
            y = int(im_id.split('_')[1])
            # this is the old code chunk
            self.or_super_class_names[eval_type] += ['/'.join(im_path.split('/')[1:3])]
            self.or_super_ys[eval_type] += [self.class_names_dict['/'.join(im_path.split('/')[1:3])]]
            self.or_im_paths[eval_type] += [os.path.join(root, im_path)]
            self.or_ys[eval_type] += [y]
            self.or_I[eval_type] += [or_I_[eval_type]]
            or_I_[eval_type] += 1

        nb_samples_counted = len(self.or_im_paths['train']) + len(self.or_im_paths['gallery']) + len(self.or_im_paths['query'])
        assert nb_samples_counted == nb_samples

        # verify that labels are sorted for next step
        # self.or_ys['query'] == sorted(self.or_ys['query'])
        # self.or_ys['gallery'] == sorted(self.or_ys['gallery'])

        assert len(self.or_ys['train']) == 25882
        assert len(self.or_ys['query']) == 14218
        assert len(self.or_ys['gallery']) == 12612

        # verify that query and gallery have same labels
        assert set(self.or_ys['query']) == set(self.or_ys['gallery'])

        # labels of query and gallery are like [1, 1, 7, 7, 8, 11, ...]
        # condense them such that ordered without spaces,
        # i.e. 1 -> 1, 7 -> 2, ...
        # same thing for train labels
        idx_to_class = {idx: i for i, idx in enumerate(
            sorted(set(self.or_ys['train']))
        )}
        self.or_ys['train'] = list(
            map(lambda x: idx_to_class[x], self.or_ys['train']))

        # TODO: I change query, gallery labels to be incremental from training labels
        idx_to_class = {idx: i + len(set(self.or_ys['train'])) for i, idx in enumerate(
            sorted(set(self.or_ys['query']))
        )}
        for _type in ['query', 'gallery']:
            self.or_ys[_type] = list(
                map(lambda x: idx_to_class[x], self.or_ys[_type]))

        # should be 3997 classes for training, 3985 for query/gallery
        assert len(set(self.or_ys['train'])) == 3997
        assert len(set(self.or_ys['query'])) == 3985
        assert len(set(self.or_ys['gallery'])) == 3985

        ##start your dataset selection here for train, val and trainval
        self.im_paths = []
        self.ys = []
        self.I = []
        self.super2ys = {}

        if dset_type == 'train':
            for ix in range(len(self.or_ys[dset_type])):
                y = self.or_ys[dset_type][ix]
                im_path = self.or_im_paths[dset_type][ix]
                ii = self.or_I[dset_type][ix]
                s_y = self.or_super_ys[dset_type][ix]
                if y not in self.super2ys.keys():
                    self.super2ys[y] = s_y
                if y in classes:
                    self.im_paths.append(im_path)
                    self.ys.append(y)
                    self.I.append(ii)

            # noisy data injection 5% mislabelled data
            np.random.seed(seed)
            self.noisy_indices = np.random.choice(self.I, int(mislabel_percentage * len(self.I)), replace=False)
            for ind in self.noisy_indices:
                orig_y = self.ys[ind]
                if orig_y + 5 > max(self.classes):
                    self.ys[ind] = orig_y - 5
                else:
                    self.ys[ind] = orig_y + 5  # cannot exceeds the training classes range


        elif dset_type in ['query', 'gallery']:
            for ix in range(len(self.or_ys[dset_type])):
                y = self.or_ys[dset_type][ix]
                im_path = self.or_im_paths[dset_type][ix]
                ii = self.or_I[dset_type][ix]
                s_y = self.or_super_ys[dset_type][ix]
                if y not in self.super2ys.keys():
                    self.super2ys[y] = s_y
                if y in classes:
                    self.im_paths.append(im_path)
                    self.ys.append(y)
                    self.I.append(ii)

        elif dset_type == 'all': # query + gallery
            for ix in range(len(self.or_ys['query'])):
                y = self.or_ys['query'][ix]
                im_path = self.or_im_paths['query'][ix]
                ii = self.or_I['query'][ix]
                s_y = self.or_super_ys['query'][ix]
                if y not in self.super2ys.keys():
                    self.super2ys[y] = s_y
                if y in classes:
                    self.im_paths.append(im_path)
                    self.ys.append(y)
                    self.I.append(ii)

            for ix in range(len(self.or_ys['gallery'])):
                y = self.or_ys['gallery'][ix]
                im_path = self.or_im_paths['gallery'][ix]
                ii = self.or_I['gallery'][ix]
                s_y = self.or_super_ys['gallery'][ix]
                if y not in self.super2ys.keys():
                    self.super2ys[y] = s_y
                if y in classes:
                    self.im_paths.append(im_path)
                    self.ys.append(y)
                    self.I.append(ii)


class InShop_hdf5(BaseDataset_hdf5):
    def __init__(self, root, source, classes, transform = None, dset_type='train'):
        BaseDataset_hdf5.__init__(self, root, source, classes, transform, prefix=dset_type + '_')

        index = 0
        self.data_y = h5py.File(root, 'r')
        #self.all_labels = torch.Tensor(self.data_y['y']).squeeze().long()
        #self.data_y.close()
        #self.data_y = None

        #for ix in tqdm(range(len(self.all_labels))):
            #print(ix, self.all_labels[ix], self.all_labels[ix].item() in self.classes, self.all_labels[ix].item())
        for ix in range(len(self.data_y[self.prefix + 'y'])):
            curr_label = self.data_y[self.prefix + 'y'][ix].item()
            if dset_type == 'train':
                if curr_label in self.classes:
                    self.ys += [curr_label]
                    self.I += [ix]
                    index += 1
            else:
                self.ys += [curr_label]
                self.I += [ix]
                index += 1

        self.data_y.close()


