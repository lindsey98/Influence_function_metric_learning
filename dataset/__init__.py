
from .cars import Cars, Cars_hdf5
from .cub import CUBirds, CUBirds_hdf5, CUBirds_class, CUBirds_hdf5_alt, CUBirds_hdf5_bb
from .sop import SOProducts, SOProducts_hdf5
from .inshop import InShop, InShop_hdf5
from .logo2k import Logo2k, Logo2k_class, Logo2k_super
from .vggface import VggFace
from . import utils


import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print(rlimit)
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))

_type = {
    'cars': Cars,
    'cars_h5': Cars_hdf5,
    'cub': CUBirds,
    'cub_h5': CUBirds_hdf5,
    'cub_class' : CUBirds_class,
    'sop': SOProducts,
    'sop_h5': SOProducts_hdf5,
    'sop_h5_mod': SOProducts_hdf5,
    'inshop': InShop,
    'inshop_h5': InShop_hdf5,
    'logo2k': Logo2k,
    'logo2k_super100': Logo2k_super,
    'logo2k_super500': Logo2k_super,
    'vgg': VggFace,
}


def load(name, root, source, classes, transform = None):
    return _type[name](root = root, source = source, classes = classes, transform = transform)

def load_inshop(name, root, source, classes, transform = None, dset_type='train'):
    return _type[name](root = root, source = source, classes = classes, transform = transform, dset_type = dset_type)

