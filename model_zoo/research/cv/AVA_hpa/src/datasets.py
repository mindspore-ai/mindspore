# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""hpa dataset"""

import os
from collections import Counter
from PIL import Image
import numpy as np
import pandas as pd
import mindspore.dataset.vision.py_transforms as transforms
from mindspore.dataset import GeneratorDataset
from mindspore.dataset.transforms.py_transforms import Compose
from src.RandAugment import RandAugment

# split train val test = 4:1:5
def split_train_val_test(sids):
    np.random.seed(286501567)
    np.random.shuffle(sids)
    ts = int(len(sids) * 0.4)
    vs = int(len(sids) * 0.5)

    return sids[:ts], sids[ts:vs], sids[vs:]


class TransformOnImg:
    """transform on image"""
    def __init__(self, mode):
        self.mode = mode
        rand_augment = RandAugment(n=2, m=10)
        self.trsfm_basic = Compose([
            transforms.ToPIL(),
            transforms.Resize(256),
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            transforms.RandomColorAdjust(0.4, 0.4, 0.4, 0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.trsfm_aux = Compose([
            transforms.ToPIL(),
            transforms.Resize(256),
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            rand_augment,
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.trsfm_train = Compose([
            transforms.ToPIL(),
            transforms.Resize(256),
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.trsfm = Compose([
            transforms.ToPIL(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def __call__(self, img, use_aux=False):
        if self.mode == "pretrain":
            if use_aux:
                img = self.trsfm_aux(img)
            else:
                img = self.trsfm_basic(img)
        else:
            if self.mode == "train:":
                img = self.trsfm_train(img)
            else:
                img = self.trsfm(img)
        return img


class BagDataCollatePretrain():
    """collect bag data on pretrain stage"""
    def __init__(self):
        pass

    def __call__(self, batch):
        imgs_basic1, imgs_basic2, imgs_aux, anns = batch
        return imgs_basic1, imgs_basic2, imgs_aux, anns


class BagDataCollate():
    """collect bag data on training stage"""
    def __init__(self, mode, max_bag_size=1):
        self.mode = mode
        self.max_bag_size = max_bag_size

    def aggregate(self, img, label, nslice):
        """aggregate data"""
        nb, _, c, h, w = img.shape
        _, nclasses = label.shape

        allimgs = np.zeros([self.max_bag_size * len(nslice), c, h, w])
        alllabels = np.zeros([self.max_bag_size * len(nslice), nclasses])

        # return images after padding
        cur = 0
        for s in range(nb):
            allimgs[cur:cur + nslice[s]] = img[s, :nslice[s], :]
            alllabels[cur:cur + nslice[s]] = label[s, :]
            cur = cur + nslice[s]

        if self.mode == "train":
            return allimgs.astype(np.float32), alllabels.astype(np.float32), nslice.astype(np.int32)

        # need `nslice` to recover bag when eval
        return allimgs.astype(np.float32), label.astype(np.float32), nslice.astype(np.int32)

    def __call__(self, batch):
        # bag of one batch
        bsid, bimgs, blabel = batch
        size = len(bsid)

        # calculate num of patch per bag
        nslice = [x.shape[0] for x in bimgs]
        max_slice = max(nslice)

        # unify the num of patch to maximum by padding
        pad_imgs = []
        for i in range(size):
            pad_img = np.pad(
                bimgs[i],
                [(0, max_slice - nslice[i]), (0, 0), (0, 0), (0, 0)],
                mode='constant',
                constant_values=0
            )
            pad_imgs.append(pad_img)

        # resort bags in a batch by the num of patch to make balance
        nslice = np.array(nslice)
        order = balance_split(nslice)

        bsid = np.array(bsid)[order]
        pad_imgs = np.array(pad_imgs)[order]
        blabel = np.array(blabel)[order]
        nslice = nslice[order]
        return self.aggregate(np.array(pad_imgs), np.array(blabel), np.array(nslice))


# balance operation
def find_i_j_v(seq):
    '''isOk[i][j][v]: find j numbers from front i sum to v
    for seq, index starts from 0
    for isOk mark, index starts from 1
    '''
    n = len(seq)
    tot = np.sum(seq)

    isOk = np.zeros((n + 1, n + 1, tot + 1), dtype=int)
    isOk[:, 0, 0] = 1

    for i in range(1, n + 1):
        jmax = min(i, n // 2)
        for j in range(1, jmax + 1):
            for v in range(1, tot // 2 + 1):
                if isOk[i - 1][j][v]:
                    isOk[i][j][v] = 1

            for v in range(1, tot // 2 + 1):
                if v >= seq[i - 1]:
                    if isOk[i - 1][j - 1][v - seq[i - 1]]:
                        isOk[i][j][v] = 1
    return isOk


def balance_split(seq):
    '''split seq to 2 sub list with equal length, sum nearly equal '''
    n = len(seq)
    tot = np.sum(seq)
    res = find_i_j_v(seq)

    i = n
    j = n // 2
    v = tot // 2

    sel_idx = []
    sel_val = []

    while not res[i][j][v] and v > 0:
        v = v - 1

    while len(sel_idx) < n // 2 and i >= 0:
        if res[i][j][v] and res[i - 1][j - 1][v - seq[i - 1]]:
            sel_idx.append(i - 1)
            sel_val.append(seq[i - 1])
            j = j - 1
            v = v - seq[i - 1]
            i = i - 1
        else:
            i = i - 1

    left = sel_idx
    right = [x for x in list(range(n)) if x not in left]
    return np.array(left + right)


class HPADataset:
    """hpa dataset"""
    def __init__(self, data_dir, mode, batch_size, bag_size, classes=10, shuffle=False):
        self.collate = BagDataCollate(mode=mode, max_bag_size=bag_size)
        self.collate_pretrain = BagDataCollatePretrain()
        self.nclasses = classes
        self.transform = TransformOnImg(mode=mode)
        self.mode = mode
        self.data_dir = data_dir
        self.bag_size = bag_size
        self.batch_size = batch_size
        filter_d, self.top_cv = self.filter_top_cv(classes)
        sids = np.array(list(sorted(filter_d.keys())))
        train_sids, val_sids, test_sids = split_train_val_test(sids)

        if mode == 'pretrain':
            self.db, self.sids = self.load_data(filter_d, train_sids, max_bag_size=bag_size)  # max_bag_size=1
        elif mode == 'train':
            self.db, self.sids = self.load_data(filter_d, train_sids, max_bag_size=bag_size)  # max_bag_size=1
        elif mode == 'val':
            self.db, self.sids = self.load_data(filter_d, val_sids, max_bag_size=bag_size)  # max_bag_size=20
        elif mode == 'test':
            self.db, self.sids = self.load_data(filter_d, test_sids, max_bag_size=bag_size)  # max_bag_size=20
        if shuffle:
            np.random.shuffle(self.sids)

    def load_data(self, d, sids, max_bag_size):
        '''
            final dict format:
            {
                "protein_id":{
                    img: ["686_A3_2_blue_red_green_1.jpg", "686_A3_2_blue_red_green_2.jpg", ...]
                    label: ['2', '5', '6']
                }
                ...
            }
        '''
        # make dict
        imgdir = self.data_dir
        final_d = {}
        for sid in sids:

            # read all images in this dir
            gene_imgs = []
            for gene_img in os.listdir(os.path.join(imgdir, sid)):
                img_pth = os.path.join(imgdir, sid, gene_img)
                gene_imgs.append(img_pth)

            # split when more than max_bag_size images
            gene_imgs = list(set(gene_imgs))
            bag_size = len(gene_imgs)
            while bag_size > max_bag_size:
                bag_size = bag_size // 2

            # save data of full-size bag
            num_bags = len(gene_imgs) // bag_size
            for i in range(num_bags):
                bag_img = gene_imgs[i * bag_size: (i + 1) * bag_size]
                gene_name = '%s_%d' % (sid, i)

                final_d[gene_name] = {}
                final_d[gene_name]['img'] = bag_img
                final_d[gene_name]['label'] = d[sid]

            # save data of none full-size bag
            if len(gene_imgs) > num_bags * bag_size:
                bag_img = gene_imgs[num_bags * bag_size:]
                gene_name = '%s_%d' % (sid, num_bags)

                final_d[gene_name] = {}
                final_d[gene_name]['img'] = bag_img
                final_d[gene_name]['label'] = d[sid]

        return final_d, sorted(final_d.keys())

    def get_sid_label(self, sid):
        sid_anns = self.db[sid]['label']
        anns = np.zeros(self.nclasses)
        for ann in sid_anns:
            anns[self.top_cv.index(ann)] = 1
        return anns

    def filter_top_cv(self, k=10, csv_file="enhanced.csv"):
        """get top k frequent labels"""
        # get label
        all_cv = []
        label_file = pd.read_csv(csv_file)
        labels = label_file['label']
        for label in labels:
            all_cv += list(label.split(";"))

        # count label num to get top-frequent label
        count = Counter(all_cv)
        top_cv = [x[0] for x in count.most_common(k)]

        # make dict mapping dir to label
        d = {}
        genes = label_file['Gene']
        labels = label_file['label']
        length = len(genes)
        for i in range(length):
            d[genes[i]] = list(labels[i].split(";"))

        filter_d = {}
        all_sids = sorted(d.keys())
        for sid in all_sids:
            for label in d[sid]:
                if label not in top_cv:
                    continue
                if sid not in filter_d:
                    filter_d[sid] = []
                filter_d[sid].append(label)

        if len(top_cv) < k:
            print("Error: top cv less than k", count)
        return filter_d, top_cv

    def __len__(self):

        return len(self.sids) // self.batch_size

    def __getitem__(self, index):
        if self.mode == "pretrain":
            imgs_basic1 = []
            imgs_basic2 = []
            imgs_aux = []
            anns = []
            for idx in range(index * self.batch_size, (index + 1) * self.batch_size):
                sid = self.sids[idx]
                sid_imgs = self.db[sid]['img']
                ann = self.get_sid_label(sid)
                for imgpth in sid_imgs:
                    img = Image.open(imgpth).convert('RGB')
                    img = np.asarray(img)
                    img_basic1 = self.transform(img)
                    img_basic2 = self.transform(img)
                    imgs_basic1.append(img_basic1)
                    imgs_basic2.append(img_basic2)
                    img_aux = self.transform(img, use_aux=True)
                    imgs_aux.append(img_aux)
                    anns.append(ann)

            imgs_basic1 = np.stack(imgs_basic1).astype(np.float32)
            imgs_basic2 = np.stack(imgs_basic2).astype(np.float32)
            imgs_aux = np.stack(imgs_aux).astype(np.float32)
            anns = np.stack(anns).astype(np.int32)
            n_b, _, n_c, n_w, n_l = imgs_basic1.shape
            imgs_basic1 = imgs_basic1.reshape((n_b, n_c, n_w, n_l))
            imgs_basic2 = imgs_basic2.reshape((n_b, n_c, n_w, n_l))
            imgs_aux = imgs_aux.reshape((n_b, n_c, n_w, n_l))
            batch = (imgs_basic1, imgs_basic2, imgs_aux, anns)
            return self.collate_pretrain(batch)

        imgs_tuple = []
        anns_tuple = []
        sids_tuple = []

        for idx in range(index * self.batch_size, (index + 1) * self.batch_size):
            imgs = []

            sid = self.sids[idx]
            sid_imgs = self.db[sid]['img']
            ann = self.get_sid_label(sid)

            for imgpth in sid_imgs:
                img = Image.open(imgpth).convert('RGB')
                img = np.asarray(img)
                img = self.transform(img)
                # transform return a tuple of length 1, tuple[0] has the image of shape 3,224,224
                imgs.append(img[0])

            imgs = np.stack(imgs)
            imgs_tuple.append(imgs)
            anns_tuple.append(ann)
            sids_tuple.append(sid)

        batch = (tuple(sids_tuple), tuple(imgs_tuple), tuple(anns_tuple))
        return self.collate(batch)


def makeup_pretrain_dataset(data_dir, batch_size, bag_size, epoch=1, shuffle=False, classes=10, num_parallel_workers=4):
    pretrain_dataset = HPADataset(data_dir=data_dir, mode="pretrain", batch_size=batch_size, bag_size=bag_size,
                                  shuffle=shuffle, classes=classes)
    ds = GeneratorDataset(pretrain_dataset, ['img_basic1', 'img_basic2', 'img_aux', 'label'],
                          num_parallel_workers=num_parallel_workers)
    return ds


def makeup_dataset(data_dir, mode, batch_size, bag_size, epoch=1, shuffle=False, classes=10, num_parallel_workers=4):
    dataset = HPADataset(data_dir=data_dir, mode=mode, batch_size=batch_size, bag_size=bag_size, shuffle=shuffle,
                         classes=classes)
    if mode == "train":
        ds = GeneratorDataset(dataset, ['imgs', 'labels', 'nslice'], num_parallel_workers=num_parallel_workers)
    else:
        ds = GeneratorDataset(dataset, ['imgs', 'labels', 'nslice'], num_parallel_workers=num_parallel_workers)
    return ds
