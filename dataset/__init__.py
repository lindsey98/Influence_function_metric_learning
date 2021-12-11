
from .cars import Cars, Cars_hdf5
from .cub import CUBirds, CUBirds_hdf5, CUBirds_class, CUBirds_hdf5_alt, CUBirds_hdf5_bb
from .sop import SOProducts, SOProducts_hdf5, SOProductsMod
from .inshop import InShop, InShop_hdf5, InShopAll
from .logo2k import Logo2k, Logo2k_class
# from .vggface import VggFace
from . import utils

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print(rlimit)
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))

_type = {
    'cars': Cars,
    'cars+179_178': Cars,
    'cars+103_102': Cars,
    'cars+183_182': Cars,
    'cars+111_179': Cars,
    'cars+139_137': Cars,
    'cub': CUBirds,
    'cub+143_145': CUBirds,
    'cub+172_178': CUBirds,
    'cub+144_142': CUBirds,
    'cub+196_192': CUBirds,
    'cub+116_114': CUBirds,
    'cub_class' : CUBirds_class,
    'sop': SOProducts,
    'sop+12659_12660': SOProductsMod,
    'sop+11419_15016': SOProductsMod,
    'sop+11855_11756': SOProductsMod,
    'sop+11835_11842': SOProductsMod,
    'sop+21626_15606': SOProductsMod,
    'inshop': InShop,
    'inshop+6070_7581': InShop,
    'inshop+5077_5589': InShop,
    'inshop+4880_5589': InShop,
    'inshop+7841_5589': InShop,
    'inshop+7403_5589': InShop,
    'inshop_all': InShopAll
}


def load(name, root, source, classes, transform = None):
    return _type[name](root = root, source = source, classes = classes, transform = transform)

def load_inshop(name, root, source, classes, transform = None, dset_type='train'):
    return _type[name](root = root, source = source, classes = classes, transform = transform, dset_type = dset_type)

