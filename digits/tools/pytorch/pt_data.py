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


logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

class LMDB_Loader(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        
        #TODO: set-up the given LMDB file (understand how LMDB works, what output does it do)
        self.transform = transform
        self.target_transform = target_transform
        self.db_path= db_path

        self.lmdb_env = lmdb.open(self.db_path, readonly=True)

        with self.lmdb_env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
            self.keys = [key for key, _ in txn.cursor()]

    def __getitem__(self, index):

        #TODO: get item from lmdb file
        datum = caffe.proto.caffe_pb2.Datum()
        lmdb_cursor = self.lmdb_txn.cursor()
        key_index ='{:08}'.format(index)
        value = lmdb_cursor.get(keys[index])
        datum.ParseFromString(value)
        
        label = datum.label
        data = caffe.io.datum_to_array(datum)

        # TODO: convert to tensor
        data = torchvision.transforms.ToTensor(data)
        label = torchvision.transforms.ToTensor(label)

        #TODO: do transforms for image
        if self.transform is not None:
            img = self.transform(data)

        return img, label

    def __len__(self):
        return self.length
    #TODO: define size of data can be len(self.imgs)
