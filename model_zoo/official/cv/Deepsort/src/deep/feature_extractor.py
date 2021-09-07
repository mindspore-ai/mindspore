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
import numpy as np
import cv2
import mindspore

from mindspore.train.serialization import load_checkpoint, load_param_into_net
from .original_model import Net

class Extractor:
    def __init__(self, model_path, batch_size=32):
        self.net = Net(reid=True)
        self.batch_size = batch_size
        param_dict = load_checkpoint(model_path)
        load_param_into_net(self.net, param_dict)
        self.size = (64, 128)

    def statistic_normalize_img(self, img, statistic_norm=True):
        """Statistic normalize images."""
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        if statistic_norm:
            img = (img - mean) / std
        img = img.astype(np.float32)
        return img

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)
        im_batch = []
        for im in im_crops:
            im = _resize(im, self.size)
            im = self.statistic_normalize_img(im)
            im = mindspore.Tensor.from_numpy(im.transpose(2, 0, 1).copy())
            im = mindspore.ops.ExpandDims()(im, 0)
            im_batch.append(im)

        im_batch = mindspore.ops.Concat(axis=0)(tuple(im_batch))
        return im_batch


    def __call__(self, im_crops):
        out = np.zeros((len(im_crops), 128), np.float32)
        num_batches = int(len(im_crops)/self.batch_size)
        s, e = 0, 0
        for i in range(num_batches):
            s, e = i * self.batch_size, (i + 1) * self.batch_size
            im_batch = self._preprocess(im_crops[s:e])
            feature = self.net(im_batch)
            out[s:e] = feature.asnumpy()
        if e < len(out):
            im_batch = self._preprocess(im_crops[e:])
            feature = self.net(im_batch)
            out[e:] = feature.asnumpy()
        return out
