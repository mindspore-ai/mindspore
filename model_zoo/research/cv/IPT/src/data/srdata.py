"""srdata"""
# Copyright 2021 Huawei Technologies Co., Ltd

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
import os
import glob
import random
import pickle
import numpy as np
import imageio
from src.data import common


def search(root, target="JPEG"):
    """srdata"""
    item_list = []
    items = os.listdir(root)
    for item in items:
        path = os.path.join(root, item)
        if os.path.isdir(path):
            item_list.extend(search(path, target))
        elif path.split('/')[-1].startswith(target):
            item_list.append(path)
        elif target in (path.split('/')[-2], path.split('/')[-3], path.split('/')[-4]):
            item_list.append(path)
        else:
            item_list = []
    return item_list


def search_dehaze(root, target="JPEG"):
    """srdata"""
    item_list = []
    items = os.listdir(root)
    for item in items:
        path = os.path.join(root, item)
        if os.path.isdir(path):
            extend_list = search_dehaze(path, target)
            if extend_list is not None:
                item_list.extend(extend_list)
        elif path.split('/')[-2].endswith(target):
            item_list.append(path)
    return item_list


class SRData():
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
            self.derain_test = os.path.join(args.dir_data, "Rain100L")
            self.derain_lr_test = search(self.derain_test, "rain")
            self.derain_hr_test = [path.replace(
                "rainy/", "no") for path in self.derain_lr_test]
        self._set_filesystem(args.dir_data)
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_hr, list_lr = self._scan()
        if args.ext.find('img') >= 0 or benchmark:
            self.images_hr, self.images_lr = list_hr, list_lr
        elif args.ext.find('sep') >= 0:
            os.makedirs(
                self.dir_hr.replace(self.apath, path_bin),
                exist_ok=True
            )
            for s in self.scale:
                if s == 1:
                    os.makedirs(
                        os.path.join(self.dir_hr),
                        exist_ok=True
                    )
                else:
                    os.makedirs(
                        os.path.join(
                            self.dir_lr.replace(self.apath, path_bin),
                            'X{}'.format(s)
                        ),
                        exist_ok=True
                    )

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

    # Below functions as used to prepare images
    def _scan(self):
        """srdata"""
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                if s != 1:
                    scale = s
                    names_lr[si].append(os.path.join(
                        self.dir_lr, 'X{}/{}x{}{}'.format(
                            s, filename, scale, self.ext[1]
                        )
                    ))
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
        if self.args.model == 'vtip' and self.args.derain and self.scale[
                self.idx_scale] == 1 and not self.args.finetune:
            norain, rain, _ = self._load_rain_test(idx)
            pair = common.set_channel(
                *[rain, norain], n_channels=self.args.n_colors)
            pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
            return pair_t[0], pair_t[1]
        if self.args.model == 'vtip' and self.args.denoise and self.scale[self.idx_scale] == 1:
            hr, _ = self._load_file_hr(idx)
            pair = self.get_patch_hr(hr)
            pair = common.set_channel(*[pair], n_channels=self.args.n_colors)
            pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
            noise = np.random.randn(*pair_t[0].shape) * self.args.sigma
            lr = pair_t[0] + noise
            lr = np.float32(np.clip(lr, 0, 255))
            return lr, pair_t[0]
        lr, hr, _ = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)

        return pair_t[0], pair_t[1]

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat

        if self.args.derain and not self.args.alltask:
            return int(len(self.derain_hr_test) / self.args.derain_test)
        return len(self.images_hr)

    def _get_index(self, idx):
        """srdata"""
        if self.train:
            return idx % len(self.images_hr)
        return idx

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

    def _load_rain(self, idx, rain_img=False):
        """srdata"""
        idx = random.randint(0, len(self.derain_img_list) - 1)
        f_lr = self.derain_img_list[idx]
        if rain_img:
            norain = imageio.imread(f_lr.replace("rainstreak", "norain"))
            rain = imageio.imread(f_lr.replace("rainstreak", "rain"))
            return norain, rain
        lr = imageio.imread(f_lr)
        return lr

    def _load_rain_test(self, idx):
        """srdata"""
        f_hr = self.derain_hr_test[idx]
        f_lr = self.derain_lr_test[idx]
        filename, _ = os.path.splitext(os.path.basename(f_lr))
        norain = imageio.imread(f_hr)
        rain = imageio.imread(f_lr)
        return norain, rain, filename

    def _load_denoise(self, idx):
        """srdata"""
        idx = self._get_index(idx)
        f_lr = self.images_hr[idx]
        norain = imageio.imread(f_lr)
        rain = imageio.imread(f_lr.replace("HR", "LR_bicubic"))
        return norain, rain

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
            hr = self.get_patch_img_hr(
                hr,
                patch_size=self.args.patch_size,
                scale=1
            )

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
                scale=scale,
                multi=(len(self.scale) > 1)
            )
            if not self.args.no_augment:
                lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        """srdata"""
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)
