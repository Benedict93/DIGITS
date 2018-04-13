import os

from PIL import Image
import os
import os.path
import sys
import caffe
import lmdb
import logging

import torch.utils.data as data
import torch
import torchvision


# Supported extensions for Loaders
DB_EXTENSIONS = {
    'hdf5': ['.H5', '.HDF5'],
    'lmdb': ['.MDB', '.LMDB'],
}
# supported image extensions
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def get_backend_of_source(db_path):
    """
    Takes a path as argument and infers the format of the data.
    If a directory is provided, it looks for the existance of an extension
    in the entire directory in an order of a priority of dbs (hdf5, lmdb, filelist, file)
    Args:
        db_path: path to a file or directory
    Returns:
        backend: the backend type
    """

    # If a directory is given, we include all its contents. Otherwise it's just the one file.
    if os.path.isdir(db_path):
        files_in_path = [fn for fn in os.listdir(db_path) if not fn.startswith('.')]
    else:
        files_in_path = [db_path]

    # Keep the below priority ordering
    for db_fmt in ['hdf5', 'lmdb']:
        ext_list = DB_EXTENSIONS[db_fmt]
        for ext in ext_list:
            if any(ext in os.path.splitext(fn)[1].upper() for fn in files_in_path):
                return db_fmt

    logging.error("Cannot infer backend from db_path (%s)." % (db_path))
    exit(-1)

class LoaderFactory(object):
    """
    A factory for data loading. It sets up a subclass with data loading
    done with the respective backend. Its output is a tensorflow queue op
    that is used to load in data, with optionally some minor postprocessing ops.
    """
    def __init__(self):
        self.backend = None
        self.db_path = None
        self.aug_dict = {}

        pass

    @staticmethod
    def set_source(db_path):
        """
        Returns the correct backend.
        """
        backend = get_backend_of_source(db_path)
        loader = None
        if backend == 'lmdb':
            loader = LMDB_Loader(db_path)
        elif backend == 'hdf5':
            loader = Hdf5Loader(db_path)
        else:
            logging.error("Backend (%s) not implemented" % (backend))
            exit(-1)
        return loader

class LMDB_Loader(data.Dataset):
    def __init__(self, lmdb_root, transform=None, target_transform=None):
        #TODO: set-up the given LMDB file (understand how LMDB works, what output does it do)
        self.transform = transform
        self.target_transform = target_transform
        self.db_path= lmdb_root

        self.lmdb_env = lmdb.open(self.db_path, readonly=True)
        self.lmdb_txn = self.lmdb_env.begin()

        self.length = self.lmdb_env.stat()['entries']

    def __getitem__(self, index):
        #TODO: get item from lmdb file
        datum = caffe.proto.caffe_pb2.Datum()
        lmdb_cursor = self.lmdb_txn.cursor()
        key_index ='{:08}'.format(index)
        value = lmdb_cursor.get(key_index)
        datum.ParseFromString(value)
        image = caffe.io.datum_to_array(datum)
        image = image.astype(np.uint8)

        label = datum.label

        # TODO: convert to tensor
        image = torchvision.transforms.ToTensor(image)
        label = torchvision.transforms.ToTensor(label)

        #TODO: do transforms for image
        if self.transform is not None:
            img = self.transform(image)

        return img, label

    def __len__(self):
        return self.length
    #TODO: define size of data can be len(self.imgs)
