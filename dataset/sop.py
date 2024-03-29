from .base import *
from tqdm import tqdm

class SOProducts(BaseDatasetMod):
    nb_train_all = 59551
    nb_test_all = 60502
    def __init__(self, root, source, classes, transform=None):
        BaseDatasetMod.__init__(self, root, source, classes, transform)

        classes_train = range(0, 11318)
        classes_test = range(11318, 22634)

        if classes.start in classes_train:
            if classes.stop - 1 in classes_train:
                train = True

        if classes.start in classes_test:
            if classes.stop - 1 in classes_test:
                train = False

        with open(
            os.path.join(
            root,
            'Ebay_{}.txt'.format('train' if train else 'test')
            )
        ) as f:

            f.readline()
            index = 0
            nb_images = 0
            self.super2ys = {}

            for (image_id, class_id, super_class_id, path) in map(str.split, f):
                nb_images += 1
                if int(class_id) - 1 in classes:
                    self.im_paths.append(os.path.join(root, path))
                    self.ys.append(int(class_id) - 1)
                    self.I += [index]
                    index += 1
                    if int(class_id) - 1 not in self.super2ys.keys():
                        self.super2ys[int(class_id) - 1] = int(super_class_id) - 1

            if train:
                assert nb_images == type(self).nb_train_all
            else:
                assert nb_images == type(self).nb_test_all

class SOProductsNoisy(BaseDatasetMod):
    nb_train_all = 59551
    nb_test_all = 60502
    def __init__(self, root, source, classes, transform=None, seed=0):
        BaseDatasetMod.__init__(self, root, source, classes, transform)

        classes_train = range(0, 11318)
        classes_test = range(11318, 22634)

        if classes.start in classes_train:
            if classes.stop - 1 in classes_train:
                train = True

        if classes.start in classes_test:
            if classes.stop - 1 in classes_test:
                train = False

        with open(
            os.path.join(
            root,
            'Ebay_{}.txt'.format('train' if train else 'test')
            )
        ) as f:

            f.readline()
            index = 0
            nb_images = 0
            self.super2ys = {}

            for (image_id, class_id, super_class_id, path) in map(str.split, f):
                nb_images += 1
                if int(class_id) - 1 in classes:
                    self.im_paths.append(os.path.join(root, path))
                    self.ys.append(int(class_id) - 1)
                    self.I += [index]
                    index += 1
                    if int(class_id) - 1 not in self.super2ys.keys():
                        self.super2ys[int(class_id) - 1] = int(super_class_id) - 1

            if train:
                assert nb_images == type(self).nb_train_all
                np.random.seed(seed)
                self.noisy_indices = np.random.choice(self.I, int(0.05 * len(self.I)), replace=False)
                for ind in self.noisy_indices:
                    orig_y = self.ys[ind]
                    if orig_y + 5 > max(self.classes):
                        self.ys[ind] = orig_y - 5
                    else:
                        self.ys[ind] = orig_y + 5  # cannot exceeds the training classes range
            else:
                assert nb_images == type(self).nb_test_all

class SOProductsMod(BaseDatasetMod):
    def __init__(self, root, source, classes, transform=None):
        BaseDatasetMod.__init__(self, root, source, classes, transform)
        index = 0; nb_images = 0
        self.super2ys = {}
        with open(os.path.join(root, 'Ebay_{}.txt'.format('train'))) as f:
            f.readline()
            for (image_id, class_id, super_class_id, path) in map(str.split, f):
                nb_images += 1
                if int(class_id) - 1 in classes:
                    self.im_paths.append(os.path.join(root, path))
                    self.ys.append(int(class_id) - 1)
                    self.I += [index]
                    index += 1
                    if int(class_id)-1 not in self.super2ys.keys():
                        self.super2ys[int(class_id) - 1] = int(super_class_id) - 1

        with open(os.path.join(root, 'Ebay_{}.txt'.format('test'))) as f:
            f.readline()
            for (image_id, class_id, super_class_id, path) in map(str.split, f):
                nb_images += 1
                if int(class_id) - 1 in classes:
                    self.im_paths.append(os.path.join(root, path))
                    self.ys.append(int(class_id) - 1)
                    self.I += [index]
                    index += 1
                    if int(class_id)-1 not in self.super2ys.keys():
                        self.super2ys[int(class_id) - 1] = int(super_class_id) - 1


class SOProducts_hdf5(BaseDataset_hdf5):
    def __init__(self, root, source, classes, transform = None):
        BaseDataset_hdf5.__init__(self, root, source, classes, transform)

        index = 0
        self.data_y = h5py.File(root, 'r')
        #self.all_labels = torch.Tensor(self.data_y['y']).squeeze().long()
        #self.data_y.close()
        #self.data_y = None

        #for ix in tqdm(range(len(self.all_labels))):
            #print(ix, self.all_labels[ix], self.all_labels[ix].item() in self.classes, self.all_labels[ix].item())
        for ix in range(len(self.data_y['y'])):
            curr_label = self.data_y['y'][ix].item()
            '''
            if self.all_labels[ix] in self.classes:
                self.ys += [self.all_labels[ix].item()]
                self.I += [ix]
                index += 1
            '''
            if curr_label in self.classes:
                self.ys += [curr_label]
                self.I += [ix]
                index += 1

        self.data_y.close()


