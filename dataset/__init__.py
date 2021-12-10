
from .cars import Cars, Cars_hdf5
from .cub import CUBirds, CUBirds_hdf5, CUBirds_class, CUBirds_hdf5_alt, CUBirds_hdf5_bb
from .sop import SOProducts, SOProducts_hdf5
from .inshop import InShop, InShop_hdf5
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
    'inshop': InShop,
}


def load(name, root, source, classes, transform = None):
    return _type[name](root = root, source = source, classes = classes, transform = transform)

def load_inshop(name, root, source, classes, transform = None, dset_type='train'):
    return _type[name](root = root, source = source, classes = classes, transform = transform, dset_type = dset_type)

