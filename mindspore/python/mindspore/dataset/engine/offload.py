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
# ==============================================================================
"""Offload Support.
"""
import json
import numpy as np

import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
import mindspore.nn as nn
import mindspore.ops.composite as C
from mindspore.ops import operations as P


def check_concat_zip_dataset(dataset):
    """
    Check if dataset is concatenated or zipped.
    """
    while dataset:
        if len(dataset.children) > 1:
            raise RuntimeError("Offload module currently does not support concatenated or zipped datasets.")
        if dataset.children:
            dataset = dataset.children[0]
            continue
        dataset = dataset.children


def apply_offload_iterators(data, offload_model):
    """
    Apply offload for non sink mode pipeline.
    """
    if len(data) != 2:
        # A temporary solution to ensure there are two columns in dataset.
        raise RuntimeError("Offload can currently only use datasets with two columns.")
    if isinstance(data[0], Tensor) is True:
        data[0] = offload_model(data[0])
    else:
        data[0] = Tensor(data[0], dtype=mstype.float32)
        data[0] = offload_model(data[0]).asnumpy()

    return data


class ApplyPreTransform(nn.Cell):
    """
    Concatenates offload model with network.
    """
    def __init__(self, transform, model):
        super(ApplyPreTransform, self).__init__(auto_prefix=False, flags=model.get_flags())
        self.transform = transform
        self.model = model

    def construct(self, x, label):
        x = self.transform(x)
        x = self.model(x, label)
        return x


class IdentityCell(nn.Cell):
    """
    Applies identity transform on given input tensors.
    """
    def __init__(self):
        super(IdentityCell, self).__init__()
        self.identity = P.Identity()

    def construct(self, x):
        return self.identity(x)


class RandomHorizontalFlip(nn.Cell):
    """
    Applies Random Horizontal Flip transform on given input tensors.
    """
    def __init__(self, prob):
        super(RandomHorizontalFlip, self).__init__()

        self.prob = Tensor(prob, dtype=mstype.float32)

        self.cast = P.Cast()
        self.shape = P.Shape()
        self.uniformReal = P.UniformReal()
        self.reshape = P.Reshape()
        self.h_flip = P.ReverseV2(axis=[2])
        self.mul = P.Mul()

    def construct(self, x):

        x = self.cast(x, mstype.float32)
        bs, h, w, c = self.shape(x)

        flip_rand_factor = self.uniformReal((bs, 1))
        flip_rand_factor = self.cast((self.prob > flip_rand_factor), mstype.float32)
        flip_rand_factor = self.reshape(C.repeat_elements(flip_rand_factor, rep=(h*w*c)), (bs, h, w, c))

        x_flip = self.h_flip(x)
        x = self.mul(x_flip, flip_rand_factor) + self.mul((1 - flip_rand_factor), x)

        return x


class RandomVerticalFlip(nn.Cell):
    """
    Applies Random Vertical Flip transform on given input tensors.
    """
    def __init__(self, prob):
        super(RandomVerticalFlip, self).__init__()

        self.prob = Tensor(prob, dtype=mstype.float32)

        self.cast = P.Cast()
        self.shape = P.Shape()
        self.uniformReal = P.UniformReal()
        self.reshape = P.Reshape()
        self.h_flip = P.ReverseV2(axis=[1])
        self.mul = P.Mul()

    def construct(self, x):

        x = self.cast(x, mstype.float32)
        bs, h, w, c = self.shape(x)

        flip_rand_factor = self.uniformReal((bs, 1))
        flip_rand_factor = self.cast((self.prob > flip_rand_factor), mstype.float32)
        flip_rand_factor = self.reshape(C.repeat_elements(flip_rand_factor, rep=(h*w*c)), (bs, h, w, c))

        x_flip = self.h_flip(x)
        x = self.mul(x_flip, flip_rand_factor) + self.mul((1 - flip_rand_factor), x)

        return x


class RandomColorAdjust(nn.Cell):
    """
    Applies Random Color Adjust transform on given input tensors.
    """
    def __init__(self, brightness, saturation):
        super(RandomColorAdjust, self).__init__()

        if isinstance(brightness, (list, tuple)):
            self.br_min = brightness[0]
            self.br_max = brightness[1]
        else:
            self.br_min = max(0, 1 - brightness)
            self.br_max = 1 + brightness

        if isinstance(saturation, (list, tuple)):
            self.sa_min = saturation[0]
            self.sa_max = saturation[1]
        else:
            self.sa_min = max(0, 1 - saturation)
            self.sa_max = 1 + saturation

        self.cast = P.Cast()
        self.shape = P.Shape()
        self.uniformReal = P.UniformReal()
        self.reshape = P.Reshape()
        self.unstack = P.Unstack(axis=-1)
        self.expand_dims = P.ExpandDims()
        self.mul = P.Mul()

    def construct(self, x):

        x = self.cast(x, mstype.float32)
        bs, h, w, c = self.shape(x)

        br_rand_factor = self.br_min + (self.br_max - self.br_min)*self.uniformReal((bs, 1))
        br_rand_factor = self.reshape(C.repeat_elements(br_rand_factor, rep=(h*w*c)), (bs, h, w, c))

        sa_rand_factor = self.sa_min + (self.sa_max - self.sa_min)*self.uniformReal((bs, 1))
        sa_rand_factor = self.reshape(C.repeat_elements(sa_rand_factor, rep=(h*w*c)), (bs, h, w, c))

        r, g, b = self.unstack(x)
        x_gray = C.repeat_elements(self.expand_dims((0.2989 * r + 0.587 * g + 0.114 * b), -1), rep=c, axis=-1)

        x = self.mul(x, br_rand_factor)
        x = C.clip_by_value(x, 0.0, 255.0)

        x = self.mul(x, sa_rand_factor) + self.mul((1 - sa_rand_factor), x_gray)
        x = C.clip_by_value(x, 0.0, 255.0)

        return x


class RandomSharpness(nn.Cell):
    """
    Applies Random Sharpness transform on given input tensors.
    """
    def __init__(self, degrees):
        super(RandomSharpness, self).__init__()

        if isinstance(degrees, (list, tuple)):
            self.degree_min = degrees[0]
            self.degree_max = degrees[1]
        else:
            self.degree_min = max(0, 1 - degrees)
            self.degree_max = 1 + degrees

        self.cast = P.Cast()
        self.shape = P.Shape()
        self.uniformReal = P.UniformReal()
        self.reshape = P.Reshape()
        self.expand_dims = P.ExpandDims()
        self.mul = P.Mul()
        self.transpose = P.Transpose()

        self.weight = np.array([[1, 1, 1], [1, 5, 1], [1, 1, 1]])/13.0
        self.weight = np.repeat(self.weight[np.newaxis, :, :], 3, axis=0)
        self.weight = np.repeat(self.weight[np.newaxis, :, :], 3, axis=0)
        self.weight = Tensor(self.weight, mstype.float32)

        self.filter = P.Conv2D(out_channel=3, kernel_size=(3, 3), pad_mode='same')

    def construct(self, x):

        x = self.cast(x, mstype.float32)
        bs, h, w, c = self.shape(x)

        degree_rand_factor = self.degree_min + (self.degree_max - self.degree_min)*self.uniformReal((bs, 1))
        degree_rand_factor = self.reshape(C.repeat_elements(degree_rand_factor, rep=(h*w*c)), (bs, h, w, c))

        x_sharp = self.filter(self.transpose(x, (0, 3, 1, 2)), self.weight)
        x_sharp = self.transpose(x_sharp, (0, 2, 3, 1))

        x = self.mul(x, degree_rand_factor) + self.mul((1 - degree_rand_factor), x_sharp)
        x = C.clip_by_value(x, 0.0, 255.0)

        return x


class Rescale(nn.Cell):
    """
    Applies Rescale transform on given input tensors.
    """
    def __init__(self, rescale, shift):
        super(Rescale, self).__init__()

        self.rescale = Tensor(rescale, dtype=mstype.float32)
        self.shift = Tensor(shift, dtype=mstype.float32)

        self.cast = P.Cast()
        self.mul = P.Mul()

    def construct(self, x):

        x = self.cast(x, mstype.float32)
        x = x * self.rescale + self.shift

        return x


class HwcToChw(nn.Cell):
    """
    Applies Channel Swap transform on given input tensors.
    """
    def __init__(self):
        super(HwcToChw, self).__init__()
        self.trans = P.Transpose()

    def construct(self, x):
        return self.trans(x, (0, 3, 1, 2))


class Normalize(nn.Cell):
    """
    Applies Normalize transform on given input tensors.
    """
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = Tensor(mean, mstype.float32)
        self.std = Tensor(std, mstype.float32)
        self.sub = P.Sub()
        self.div = P.Div()
        self.cast = P.Cast()

    def construct(self, x):
        x = self.cast(x, mstype.float32)
        x = self.sub(x, self.mean)
        x = self.div(x, self.std)
        return x


class OffloadModel():
    def __init__(self, func, args_names=None):
        self.func = func
        self.args_names = args_names


# Dictionary connecting operation name to model
op_to_model = {
    "HWC2CHW": OffloadModel(HwcToChw),
    "HwcToChw": OffloadModel(HwcToChw),
    "Normalize": OffloadModel(Normalize, ["mean", "std"]),
    "RandomColorAdjust": OffloadModel(RandomColorAdjust, ["brightness", "saturation"]),
    "RandomHorizontalFlip": OffloadModel(RandomHorizontalFlip, ["prob"]),
    "RandomSharpness": OffloadModel(RandomSharpness, ["degrees"]),
    "RandomVerticalFlip": OffloadModel(RandomVerticalFlip, ["prob"]),
    "Rescale": OffloadModel(Rescale, ["rescale", "shift"])
}


class GetModelFromJson2Col(nn.Cell):
    """
    Generates offload ME model from offload JSON file for a single map op.
    """
    def __init__(self, json_offload):
        super(GetModelFromJson2Col, self).__init__()
        self.me_ops = []
        if json_offload is not None:
            offload_ops = json_offload["operations"]
            for op in offload_ops:
                name = op["tensor_op_name"]
                args = op["tensor_op_params"]
                op_model = op_to_model[name]
                op_model_inputs = []
                if op_model.args_names is not None:
                    for arg_key in op_model.args_names:
                        op_model_inputs.append(args[arg_key])

                self.me_ops.append(op_model.func(*op_model_inputs))
        else:
            raise RuntimeError("Offload hardware accelarator cannot be applied for this pipeline.")

        self.cell = nn.SequentialCell(self.me_ops)

    def construct(self, x):
        return self.cell(x)


class GetOffloadModel(nn.Cell):
    """
    Generates offload ME model.
    """
    def __init__(self, dataset_consumer):
        super(GetOffloadModel, self).__init__()
        self.transform_list = []
        json_offload = json.loads(dataset_consumer.GetOffload())
        if json_offload is not None:
            for node in json_offload:
                if node["op_type"] == 'Map':
                    self.transform_list.append(GetModelFromJson2Col(node))
            self.transform_list.reverse()

    def construct(self, x):
        for transform in self.transform_list:
            x = transform(x)
        return x
