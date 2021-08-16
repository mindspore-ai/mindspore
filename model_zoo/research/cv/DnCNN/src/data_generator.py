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

"""process the training data set"""

import os
import pickle
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128


def gen_patches(file_name):
    """get multiscale patches from a single image"""
    img = cv2.imread(file_name, 0)  # gray scale
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                x = img_scaled[i:i+patch_size, j:j+patch_size]
                for _ in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)
    return patches


def data_aug(img, mode=0):
    """data augmentation"""
    if mode == 0:
        pass
    elif mode == 1:
        img = np.flipud(img)
    elif mode == 2:
        img = np.rot90(img)
    elif mode == 3:
        img = np.flipud(np.rot90(img))
    elif mode == 4:
        img = np.rot90(img, k=2)
    elif mode == 5:
        img = np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        img = np.rot90(img, k=3)
    elif mode == 7:
        img = np.flipud(np.rot90(img, k=3))
    else:
        raise Exception("Invalid mode!", mode)
    return img


def generate_save_patches(data_dir='data/Train400', verbose=False):
    """generate image patches and save them"""
    dir_name, basename = os.path.dirname(data_dir), os.path.basename(data_dir)
    pkl_file = os.path.join(dir_name, basename+'.pkl')
    if not os.path.exists(pkl_file):
        file_list = glob.glob(data_dir + '/*.png')
        data = []
        for i, file in enumerate(file_list):
            patches = gen_patches(file)
            for patch in patches:
                data.append(patch)
            if verbose:
                print(str(i + 1) + '/' + str(len(file_list)) + ' is done ^_^')
        data = np.array(data, dtype='uint8')
        data = np.expand_dims(data, axis=3)
        discard_n = len(data) - len(data) // batch_size * batch_size  # because of batch namalization
        data = np.delete(data, range(discard_n), axis=0)  # (238336, 40, 40, 1), uint8
        # normalization and swap axis
        data = data.astype('float32') / 255.0
        data = data.transpose((0, 3, 1, 2))  # (238336, 1, 40, 40), float32
        with open(pkl_file, 'wb') as f:
            pickle.dump(data, f)
        print('^_^-training data finished-^_^')
    else:
        print('The .pkl file is prepared, loading...')
        f = open(pkl_file, 'rb')
        data = pickle.load(f)
    return data


class DenoisingDataset:
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches
        sigma: noise level, e.g., 25
    """
    def __init__(self, data_dir, sigma):
        super(DenoisingDataset, self).__init__()
        self.xs = generate_save_patches(data_dir)
        self.sigma = sigma

    def __getitem__(self, index):
        # print(self.xs.shape, index)  # (238336, 1, 40, 40)
        batch_x = self.xs[index, ...]
        noise = np.random.standard_normal(size=batch_x.shape) * (self.sigma/255.0)
        batch_y = batch_x + noise
        return batch_y.astype(np.float32), noise.astype(np.float32)

    def __len__(self):
        return len(self.xs)


def show(x, title=None, cbar=False, figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()
