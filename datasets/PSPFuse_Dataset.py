# this dataset can only be used in pytorch2
import torch.utils.data as data
from caffe.proto import caffe_pb2
from PIL import Image
import numpy as np
import caffe
import lmdb
import sys
import os
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

class FuseLmdbDataset(data.Dataset):
    def __init__(self, root, mf1_path, mf2_path, gt_path, pair_transform=None, input_transform=None, target_transform=None):
        super(FuseLmdbDataset, self).__init__()

        gt_path = os.path.join(root, gt_path)
        mf1_path = os.path.join(root, mf1_path)
        mf2_path = os.path.join(root, mf2_path)

        self.env0 = lmdb.open(gt_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.env1 = lmdb.open(mf1_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.env2 = lmdb.open(mf2_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)

        self.datum = caffe_pb2.Datum()

        with self.env0.begin(write=False) as txn:
            self.length = txn.stat()['entries']

        if sys.version_info[0] == 2:
            cache_file = '_cache_' + gt_path.replace('/', '_') + '2'
        else:
            cache_file = '_cache_' + gt_path.replace('/', '_') + '3'

        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env0.begin(write=False) as txn:
                self.keys = [key for key, _ in txn.cursor()]
            pickle.dump(self.keys, open(cache_file, "wb"))

        self.pair_transform = pair_transform
        self.input_transform = input_transform
        self.target_transform = target_transform

    def getimg(self, env, index):
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        self.datum.ParseFromString(imgbuf)
        #npimg = caffe.io.datum_to_array(self.datum)
        npimg = np.transpose(caffe.io.datum_to_array(self.datum), [2, 1, 0])
        #print(npimg.shape)
        #print(npimg.max())
        #print(npimg.dtype)
        img = Image.fromarray(npimg.astype('uint8'))

        return img

    def __getitem__(self, index):
        img1, img2, target = None, None, None
        env0, env1, env2 = self.env0, self.env1, self.env2
        img1, img2, target = self.getimg(env1, index), self.getimg(env2, index), self.getimg(env0, index)

        if self.pair_transform:
            img1, img2, target = self.pair_transform(img1, img2, target)

        if self.input_transform:
            img1 = self.input_transform(img1)
            img2 = self.input_transform(img2)

        if self.target_transform:
            target = self.target_transform(target)

        return img1, img2, target

    def __len__(self):
        return self.length
