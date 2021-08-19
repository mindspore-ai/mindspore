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
"""ImageDataLoader_3channel"""
import os
import random
import cv2
import numpy as np
import pandas as pd


class ImageDataLoader_3channel():
    def __init__(self, data_path, gt_path, shuffle=False, gt_downsample=False, pre_load=False):
        # pre_load: if true, all training and validation images are loaded into CPU RAM for faster processing.
        #          This avoids frequent file reads. Use this only for small datasets.
        self.data_path = data_path
        self.gt_path = gt_path
        self.gt_downsample = gt_downsample
        self.pre_load = pre_load
        self.data_files = [filename for filename in os.listdir(data_path) \
                           if os.path.isfile(os.path.join(data_path, filename))]
        self.data_files.sort()
        self.shuffle = shuffle
        if shuffle:
            random.seed(2468)
        self.num_samples = len(self.data_files)
        self.blob_list = {}
        self.id_list = range(0, self.num_samples)
        if self.pre_load:
            print('Pre-loading the data. This may take a while...')
            idx = 0
            for fname in self.data_files:

                img = cv2.imread(os.path.join(self.data_path, fname), 0)
                img = img.astype(np.float32, copy=False)
                ht = img.shape[0]
                wd = img.shape[1]
                ht_1 = (ht // 4) * 4
                wd_1 = (wd // 4) * 4
                img = cv2.resize(img, (wd_1, ht_1))

                img = img.reshape((1, img.shape[0], img.shape[1]))
                den = pd.read_csv(os.path.join(self.gt_path, os.path.splitext(fname)[0] + '.csv'), sep=',',
                                  header=None).values
                den = den.astype(np.float32, copy=False)
                if self.gt_downsample:
                    # print(den.shape)
                    wd_1 = wd_1 // 4
                    ht_1 = ht_1 // 4
                    den = cv2.resize(den, (wd_1, ht_1))
                    den = den * ((wd * ht) / (wd_1 * ht_1))
                else:
                    den = cv2.resize(den, (wd_1, ht_1))
                    den = den * ((wd * ht) / (wd_1 * ht_1))

                den = den.reshape((1, den.shape[0], den.shape[1]))
                blob = {}
                blob['data'] = img
                blob['gt_density'] = den
                blob['fname'] = fname
                self.blob_list[idx] = blob
                idx = idx + 1
                if idx % 100 == 0:
                    print('Loaded ', idx, '/', self.num_samples, 'files')

            print('Completed Loading ', idx, 'files')

    def __iter__(self):
        if self.shuffle:
            if self.pre_load:
                random.shuffle(list(self.id_list))
            else:
                random.shuffle(list(self.data_files))
        files = self.data_files
        id_list = self.id_list

        for idx in id_list:
            if self.pre_load:
                blob = self.blob_list[idx]
                blob['idx'] = idx
            else:
                fname = files[idx]
                img = cv2.imread(os.path.join(self.data_path, fname), 0)
                img = img.astype(np.float32, copy=False)
                ht = img.shape[0]
                wd = img.shape[1]
                ht_1 = (ht / 4) * 4
                wd_1 = (wd / 4) * 4
                img = cv2.resize(img, (wd_1, ht_1))

                img = img.reshape((1, img.shape[0], img.shape[1]))
                den = pd.read_csv(os.path.join(self.gt_path, os.path.splitext(fname)[0] + '.csv'), sep=',',
                                  header=None).as_matrix()
                den = den.astype(np.float32, copy=False)

                if self.gt_downsample:
                    wd_1 = wd_1 / 4
                    ht_1 = ht_1 / 4
                    den = cv2.resize(den, (wd_1, ht_1))
                    den = den * ((wd * ht) / (wd_1 * ht_1))
                else:
                    den = cv2.resize(den, (wd_1, ht_1))
                    den = den * ((wd * ht) / (wd_1 * ht_1))

                den = den.reshape((1, den.shape[0], den.shape[1]))
                blob = {}
                blob['data'] = img
                blob['gt_density'] = den
                blob['fname'] = fname

            yield blob

    def get_num_samples(self):
        return self.num_samples
