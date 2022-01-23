
from .cars import Cars, Cars_hdf5, CarsNoisy
from .cub import CUBirds, CUBirds_hdf5, CUBirds_class, CUBirds_hdf5_alt, CUBirds_hdf5_bb, CUBirdsRemoval, CUBirdsNoisy
from .sop import SOProducts, SOProducts_hdf5, SOProductsMod, SOProductsNoisy
from .inshop import InShop, InShop_hdf5, InShopNoisy
# from .vggface import VggFace
from . import utils

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print(rlimit)
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))

_type = {
    'cars': Cars,
    'cub': CUBirds,
    'sop': SOProducts,
    'inshop': InShop,
    'cub_noisy': CUBirdsNoisy,
    'cars_noisy': CarsNoisy,
    'inshop_noisy': InShopNoisy,
    'sop_noisy': SOProductsNoisy

}

def load_noisy(name, root, source, classes, transform = None, seed=0):
    return _type[name](root = root, source = source, classes = classes, transform = transform, seed=seed)

def load_noisy_inshop(name, root, source, classes, transform = None, dset_type='train', seed=0):
    return _type[name](root=root, source=source, classes=classes,
                       transform=transform,
                       dset_type=dset_type,
                       seed=seed)

def load(name, root, source, classes, transform = None):
    return _type[name](root = root, source = source, classes = classes, transform = transform)

def load_inshop(name, root, source, classes, transform = None, dset_type='train'):
    return _type[name](root = root, source = source, classes = classes,
                       transform = transform, dset_type = dset_type)

