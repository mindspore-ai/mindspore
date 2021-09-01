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
"""Metric for evaluation."""
import os
import math

from PIL import Image
import numpy as np
from mindspore import nn, Tensor, ops
from mindspore import dtype as mstype
from mindspore.ops.operations.comm_ops import ReduceOp

try:
    from model_utils.device_adapter import get_rank_id, get_device_num
except ImportError:
    get_rank_id = None
    get_device_num = None
finally:
    pass


class SelfEnsembleWrapperNumpy:
    """
    SelfEnsembleWrapperNumpy using numpy
    """

    def __init__(self, net):
        super(SelfEnsembleWrapperNumpy, self).__init__()
        self.net = net

    def hflip(self, x):
        return x[:, :, :, ::-1]

    def vflip(self, x):
        return x[:, :, ::-1, :]

    def trnsps(self, x):
        return x.transpose(0, 1, 3, 2)

    def aug_x8(self, x):
        """
        do x8 augments for input image
        """
        # hflip
        hx = self.hflip(x)
        # vflip
        vx = self.vflip(x)
        vhx = self.vflip(hx)
        # trnsps
        tx = self.trnsps(x)
        thx = self.trnsps(hx)
        tvx = self.trnsps(vx)
        tvhx = self.trnsps(vhx)
        return x, hx, vx, vhx, tx, thx, tvx, tvhx

    def aug_x8_reverse(self, x, hx, vx, vhx, tx, thx, tvx, tvhx):
        """
        undo x8 augments for input images
        """
        # trnsps
        tvhx = self.trnsps(tvhx)
        tvx = self.trnsps(tvx)
        thx = self.trnsps(thx)
        tx = self.trnsps(tx)
        # vflip
        tvhx = self.vflip(tvhx)
        tvx = self.vflip(tvx)
        vhx = self.vflip(vhx)
        vx = self.vflip(vx)
        # hflip
        tvhx = self.hflip(tvhx)
        thx = self.hflip(thx)
        vhx = self.hflip(vhx)
        hx = self.hflip(hx)
        return x, hx, vx, vhx, tx, thx, tvx, tvhx

    def to_numpy(self, *inputs):
        if inputs:
            return None
        if len(inputs) == 1:
            return inputs[0].asnumpy()
        return [x.asnumpy() for x in inputs]

    def to_tensor(self, *inputs):
        if inputs:
            return None
        if len(inputs) == 1:
            return Tensor(inputs[0])
        return [Tensor(x) for x in inputs]

    def set_train(self, mode=True):
        self.net.set_train(mode)
        return self

    def __call__(self, x):
        x = self.to_numpy(x)
        x0, x1, x2, x3, x4, x5, x6, x7 = self.aug_x8(x)
        x0, x1, x2, x3, x4, x5, x6, x7 = self.to_tensor(x0, x1, x2, x3, x4, x5, x6, x7)
        x0 = self.net(x0)
        x1 = self.net(x1)
        x2 = self.net(x2)
        x3 = self.net(x3)
        x4 = self.net(x4)
        x5 = self.net(x5)
        x6 = self.net(x6)
        x7 = self.net(x7)
        x0, x1, x2, x3, x4, x5, x6, x7 = self.to_numpy(x0, x1, x2, x3, x4, x5, x6, x7)
        x0, x1, x2, x3, x4, x5, x6, x7 = self.aug_x8_reverse(x0, x1, x2, x3, x4, x5, x6, x7)
        x0, x1, x2, x3, x4, x5, x6, x7 = self.to_tensor(x0, x1, x2, x3, x4, x5, x6, x7)
        return (x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7) / 8


class SelfEnsembleWrapper(nn.Cell):
    """
    because of [::-1] operator error, use "SelfEnsembleWrapperNumpy" instead
    """
    def __init__(self, net):
        super(SelfEnsembleWrapper, self).__init__()
        self.net = net

    def hflip(self, x):
        raise NotImplementedError("https://gitee.com/mindspore/mindspore/issues/I41ONQ?from=project-issue")

    def vflip(self, x):
        raise NotImplementedError("https://gitee.com/mindspore/mindspore/issues/I41ONQ?from=project-issue")

    def trnsps(self, x):
        return x.transpose(0, 1, 3, 2)

    def aug_x8(self, x):
        """
        do x8 augments for input image
        """
        # hflip
        hx = self.hflip(x)
        # vflip
        vx = self.vflip(x)
        vhx = self.vflip(hx)
        # trnsps
        tx = self.trnsps(x)
        thx = self.trnsps(hx)
        tvx = self.trnsps(vx)
        tvhx = self.trnsps(vhx)
        return x, hx, vx, vhx, tx, thx, tvx, tvhx

    def aug_x8_reverse(self, x, hx, vx, vhx, tx, thx, tvx, tvhx):
        """
        undo x8 augments for input images
        """
        # trnsps
        tvhx = self.trnsps(tvhx)
        tvx = self.trnsps(tvx)
        thx = self.trnsps(thx)
        tx = self.trnsps(tx)
        # vflip
        tvhx = self.vflip(tvhx)
        tvx = self.vflip(tvx)
        vhx = self.vflip(vhx)
        vx = self.vflip(vx)
        # hflip
        tvhx = self.hflip(tvhx)
        thx = self.hflip(thx)
        vhx = self.hflip(vhx)
        hx = self.hflip(hx)
        return x, hx, vx, vhx, tx, thx, tvx, tvhx

    def construct(self, x):
        """
        do x8 aug, run network, undo x8 aug, calculate mean for 8 output
        """
        x0, x1, x2, x3, x4, x5, x6, x7 = self.aug_x8(x)
        x0 = self.net(x0)
        x1 = self.net(x1)
        x2 = self.net(x2)
        x3 = self.net(x3)
        x4 = self.net(x4)
        x5 = self.net(x5)
        x6 = self.net(x6)
        x7 = self.net(x7)
        x0, x1, x2, x3, x4, x5, x6, x7 = self.aug_x8_reverse(x0, x1, x2, x3, x4, x5, x6, x7)
        return (x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7) / 8


class Quantizer(nn.Cell):
    """
    clip by [0.0, 255.0], rount to int
    """
    def __init__(self, _min=0.0, _max=255.0):
        super(Quantizer, self).__init__()
        self.round = ops.Round()
        self._min = _min
        self._max = _max

    def construct(self, x):
        x = ops.clip_by_value(x, self._min, self._max)
        x = self.round(x)
        return x


class TensorSyncer(nn.Cell):
    """
    sync metric values from all mindspore-processes
    """
    def __init__(self, _type="sum"):
        super(TensorSyncer, self).__init__()
        self._type = _type.lower()
        if self._type == "sum":
            self.ops = ops.AllReduce(ReduceOp.SUM)
        elif self._type == "gather":
            self.ops = ops.AllGather()
        else:
            raise ValueError(f"TensorSyncer._type == {self._type} is not support")

    def construct(self, x):
        return self.ops(x)


class _DistMetric(nn.Metric):
    """
    gather data from all rank while eval(True)
    _type(str): choice from ["avg", "sum"].
    """
    def __init__(self, _type):
        super(_DistMetric, self).__init__()
        self._type = _type.lower()
        self.all_reduce_sum = None
        if get_device_num is not None and get_device_num() > 1:
            self.all_reduce_sum = TensorSyncer(_type="sum")
        self.clear()

    def _accumulate(self, value):
        if isinstance(value, (list, tuple)):
            self._acc_value += sum(value)
            self._count += len(value)
        else:
            self._acc_value += value
            self._count += 1

    def clear(self):
        self._acc_value = 0.0
        self._count = 0

    def eval(self, sync=True):
        """
        sync: True, return metric value merged from all mindspore-processes
        sync: False, return metric value in this single mindspore-processes
        """
        if self._count == 0:
            raise RuntimeError('self._count == 0')
        if self.sum is not None and sync:
            data = Tensor([self._acc_value, self._count], mstype.float32)
            data = self.all_reduce_sum(data)
            acc_value, count = self._convert_data(data).tolist()
        else:
            acc_value, count = self._acc_value, self._count
        if self._type == "avg":
            return acc_value / count
        if self._type == "sum":
            return acc_value
        raise RuntimeError(f"_DistMetric._type={self._type} is not support")


class PSNR(_DistMetric):
    """
    Define PSNR metric for SR network.
    """
    def __init__(self, rgb_range, shave):
        super(PSNR, self).__init__(_type="avg")
        self.shave = shave
        self.rgb_range = rgb_range
        self.quantize = Quantizer(0.0, 255.0)

    def update(self, *inputs):
        """
        update psnr
        """
        if len(inputs) != 2:
            raise ValueError('PSNR need 2 inputs (sr, hr), but got {}'.format(len(inputs)))
        sr, hr = inputs
        sr = self.quantize(sr)
        diff = (sr - hr) / self.rgb_range
        valid = diff
        if self.shave is not None and self.shave != 0:
            valid = valid[..., self.shave:(-self.shave), self.shave:(-self.shave)]
        mse_list = (valid ** 2).mean(axis=(1, 2, 3))
        mse_list = self._convert_data(mse_list).tolist()
        psnr_list = [float(1e32) if mse == 0 else(- 10.0 * math.log10(mse)) for mse in mse_list]
        self._accumulate(psnr_list)


class SaveSrHr(_DistMetric):
    """
    help to save sr and hr
    """
    def __init__(self, save_dir):
        super(SaveSrHr, self).__init__(_type="sum")
        self.save_dir = save_dir
        self.quantize = Quantizer(0.0, 255.0)
        self.rank_id = 0 if get_rank_id is None else get_rank_id()
        self.device_num = 1 if get_device_num is None else get_device_num()

    def update(self, *inputs):
        """
        update images to save
        """
        if len(inputs) != 2:
            raise ValueError('SaveSrHr need 2 inputs (sr, hr), but got {}'.format(len(inputs)))
        sr, hr = inputs
        sr = self.quantize(sr)
        sr = self._convert_data(sr).astype(np.uint8)
        hr = self._convert_data(hr).astype(np.uint8)
        for s, h in zip(sr.transpose(0, 2, 3, 1), hr.transpose(0, 2, 3, 1)):
            idx = self._count * self.device_num + self.rank_id
            sr_path = os.path.join(self.save_dir, f"{idx:0>4}_sr.png")
            Image.fromarray(s).save(sr_path)
            hr_path = os.path.join(self.save_dir, f"{idx:0>4}_hr.png")
            Image.fromarray(h).save(hr_path)
            self._accumulate(1)
