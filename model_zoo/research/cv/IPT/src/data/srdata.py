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
"""srdata"""
import os
import glob
import random
import pickle
import numpy as np
import imageio
from src.data import common
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class SRData:
    """srdata"""
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.input_large = (args.model == 'VDSR')
        self.scale = args.scale
        self.idx_scale = 0

        if self.args.derain:
            if self.train:
                self.derain_dataroot = os.path.join(args.dir_data, "RainTrainL")
                self.clear_train = common.search(self.derain_dataroot, "norain")
                self.rain_train = []
                for path in self.clear_train:
                    change_path = path.split('/')
                    change_path[-1] = change_path[-1][2:]
                    self.rain_train.append('/'.join(change_path))
                self.derain_test = os.path.join(args.dir_data, "Rain100L")
                self.deblur_lr_test = common.search(self.derain_test, "rain")
                self.deblur_hr_test = [path.replace("rainy/", "no") for path in self.deblur_lr_test]
                self.derain_hr_test = self.deblur_hr_test
            else:
                self.derain_test = os.path.join(args.dir_data, "Rain100L")
                self.derain_lr_test = common.search(self.derain_test, "rain")
                self.derain_hr_test = [path.replace("rainy/", "no") for path in self.derain_lr_test]
        self._set_filesystem(args.dir_data)
        self._set_img(args)
        if self.args.derain and self.train:
            self.images_hr, self.images_lr = self.clear_train, self.rain_train
        if train:
            self._repeat(args)

    def _set_img(self, args):
        """srdata"""
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_hr, list_lr = self._scan()
        if args.ext.find('img') >= 0 or self.benchmark:
            self.images_hr, self.images_lr = list_hr, list_lr
        elif args.ext.find('sep') >= 0:
            os.makedirs(self.dir_hr.replace(self.apath, path_bin), exist_ok=True)
            for s in self.scale:
                if s == 1:
                    os.makedirs(os.path.join(self.dir_hr), exist_ok=True)
                else:
                    os.makedirs(
                        os.path.join(self.dir_lr.replace(self.apath, path_bin), 'X{}'.format(s)), exist_ok=True)

            self.images_hr, self.images_lr = [], [[] for _ in self.scale]
            for h in list_hr:
                b = h.replace(self.apath, path_bin)
                b = b.replace(self.ext[0], '.pt')
                self.images_hr.append(b)
                self._check_and_load(args.ext, h, b, verbose=True)
            for i, ll in enumerate(list_lr):
                for l in ll:
                    b = l.replace(self.apath, path_bin)
                    b = b.replace(self.ext[1], '.pt')
                    self.images_lr[i].append(b)
                    self._check_and_load(args.ext, l, b, verbose=True)

    def _repeat(self, args):
        """srdata"""
        n_patches = args.batch_size * args.test_every
        n_images = len(args.data_train) * len(self.images_hr)
        if n_images == 0:
            self.repeat = 0
        else:
            self.repeat = max(n_patches // n_images, 1)

    def _scan(self):
        """srdata"""
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0])))
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                if s != 1:
                    scale = s
                    names_lr[si].append(os.path.join(self.dir_lr, 'X{}/{}x{}{}' \
                        .format(s, filename, scale, self.ext[1])))
        for si, s in enumerate(self.scale):
            if s == 1:
                names_lr[si] = names_hr
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name[0])
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('.png', '.png')

    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)

    def __getitem__(self, idx):
        if self.args.derain and self.scale[self.idx_scale] == 1:
            if self.train:
                lr, hr, filename = self._load_file_deblur(idx)
                pair = self.get_patch(lr, hr)
                pair = common.set_channel(*pair, n_channels=self.args.n_colors)
                pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
            else:
                norain, rain, filename = self._load_rain_test(idx)
                pair = common.set_channel(*[rain, norain], n_channels=self.args.n_colors)
                pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
            return pair_t[0], pair_t[1], [self.idx_scale], [filename]

        if self.args.denoise and self.scale[self.idx_scale] == 1:
            hr, filename = self._load_file_hr(idx)
            pair = self.get_patch_hr(hr)
            pair = common.set_channel(*[pair], n_channels=self.args.n_colors)
            pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
            noise = np.random.randn(*pair_t[0].shape) * self.args.sigma
            lr = pair_t[0] + noise
            lr = np.float32(np.clip(lr, 0, 255))
            return lr, pair_t[0], [self.idx_scale], [filename]
        lr, hr, filename = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)

        return pair_t[0], pair_t[1], [self.idx_scale], [filename]

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat

        if self.args.derain and not self.args.alltask:
            return int(len(self.derain_hr_test) / self.args.derain_test)
        return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        return idx

    def _load_file_deblur(self, idx, train=True):
        """srdata"""
        idx = self._get_index(idx)
        if train:
            f_hr = self.images_hr[idx]
            f_lr = self.images_lr[idx]
        else:
            f_hr = self.deblur_hr_test[idx]
            f_lr = self.deblur_lr_test[idx]
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        filename = f_hr[-27:-17] + filename
        hr = imageio.imread(f_hr)
        lr = imageio.imread(f_lr)
        return lr, hr, filename

    def _load_file_hr(self, idx):
        """srdata"""
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr)
        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
        return hr, filename

    def _load_rain_test(self, idx):
        f_hr = self.derain_hr_test[idx]
        f_lr = self.derain_lr_test[idx]
        filename, _ = os.path.splitext(os.path.basename(f_lr))
        norain = imageio.imread(f_hr)
        rain = imageio.imread(f_lr)
        return norain, rain, filename

    def _load_file(self, idx):
        """srdata"""
        idx = self._get_index(idx)

        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr)
            lr = imageio.imread(f_lr)
        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)

        return lr, hr, filename

    def get_patch_hr(self, hr):
        """srdata"""
        if self.train:
            hr = self.get_patch_img_hr(hr, patch_size=self.args.patch_size, scale=1)
        return hr

    def get_patch_img_hr(self, img, patch_size=96, scale=2):
        """srdata"""
        ih, iw = img.shape[:2]

        tp = patch_size
        ip = tp // scale

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)

        ret = img[iy:iy + ip, ix:ix + ip, :]

        return ret

    def get_patch(self, lr, hr):
        """srdata"""
        scale = self.scale[self.idx_scale]
        if self.train:
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.args.patch_size * scale,
                scale=scale)
            if not self.args.no_augment:
                lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)
