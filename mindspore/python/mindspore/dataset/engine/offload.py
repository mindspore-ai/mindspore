# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
from mindspore.ops.primitive import constexpr


def check_add_offload_sink_mode(dataset, dataset_helper, network):
    """
    Check if any map operations were removed to be offloaded and apply the transforms if so.
    """
    if hasattr(dataset, '__no_send__'):
        # Dataset was not sent to device. Skip adding offload.
        return network
    # We don't use dataset.__transfer_dataset__ because there will be a device_queue rdr warning log
    iterator = dataset.create_tuple_iterator(num_epochs=1)
    if iterator.offload_model is not None:
        network = ApplyPreTransform(iterator.offload_model, network)
    return network


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


def get_col_idxs(node_cols, ds_cols):
    """
    Get the index(es) of the input column(s) from the dataset
    """
    col_idxs = []
    non_exist_cols = []
    # temporary error if multiple node columns
    if len(node_cols) > 1:
        raise RuntimeError(
            "Offload hardware accelerator currently does not support map operations with multiple input columns")
    for node_col in node_cols:
        if node_col in ds_cols:
            col_idxs.append(ds_cols.index(node_col))
        else:
            non_exist_cols.append(node_col)
    if non_exist_cols:
        raise RuntimeError(
            ("The following input column(s) for an offloaded map operation "
             "do not exist: {}").format(non_exist_cols))

    return col_idxs


def apply_offload_iterators(data, offload_model):
    """
    Apply offload for non sink mode pipeline.
    """
    non_tensor_idxs = []
    for i, _ in enumerate(data):
        if not isinstance(data[i], Tensor):
            data[i] = Tensor(data[i], dtype=mstype.float32)
            non_tensor_idxs.append(i)

    data = offload_model(data)
    data = list(data)
    for idx in non_tensor_idxs:
        data[idx] = data[idx].asnumpy()

    return data


@constexpr
def check_input_dims(x_shape, required_dim, offload_op_name):
    """
    Check if input has the required number of dimensions for the operation.
    """
    input_dim = len(x_shape)
    if input_dim is not required_dim:
        raise ValueError("For %s offload operation, the dimension of input should be %d, but got %d." %
                         (offload_op_name, required_dim, input_dim))


class ApplyPreTransform(nn.Cell):
    """
    Concatenates offload model with network.
    """

    def __init__(self, transform, model):
        super(ApplyPreTransform, self).__init__(auto_prefix=False, flags=model.get_flags())
        self.transform = transform
        self.model = model

    def construct(self, *x):
        data = []
        for data_col in x:
            data.append(data_col)

        data = self.transform(data)
        data = self.model(*data)

        return data


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
        self.reshape = P.Reshape()
        self.h_flip = P.ReverseV2(axis=[2])
        self.mul = P.Mul()

    def construct(self, x):

        x = self.cast(x, mstype.float32)
        x_shape = self.shape(x)
        check_input_dims(x_shape, 4, 'RandomHorizontalFlip')
        bs, h, w, c = x_shape

        flip_rand_factor = Tensor(np.random.uniform(size=(bs, 1)), dtype=mstype.float32)
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
        self.reshape = P.Reshape()
        self.h_flip = P.ReverseV2(axis=[1])
        self.mul = P.Mul()

    def construct(self, x):

        x = self.cast(x, mstype.float32)
        x_shape = self.shape(x)
        check_input_dims(x_shape, 4, 'RandomVerticalFlip')
        bs, h, w, c = x_shape

        flip_rand_factor = Tensor(np.random.uniform(size=(bs, 1)), dtype=mstype.float32)
        flip_rand_factor = self.cast((self.prob > flip_rand_factor), mstype.float32)
        flip_rand_factor = self.reshape(C.repeat_elements(flip_rand_factor, rep=(h*w*c)), (bs, h, w, c))

        x_flip = self.h_flip(x)
        x = self.mul(x_flip, flip_rand_factor) + self.mul((1 - flip_rand_factor), x)

        return x


class GenerateRandBatch(nn.Cell):
    """
    Generate batch with random values uniformly selected from [degree_min, degree_max].
    """

    def __init__(self):
        super(GenerateRandBatch, self).__init__()

        self.ones = P.Ones()
        self.reshape = P.Reshape()

    def __call__(self, degree_min, degree_max, check_rand, shape):

        bs, h, w, c = shape
        rand_factor = Tensor(np.random.uniform(size=(bs, 1)), dtype=mstype.float32)
        rand_factor = degree_min + (degree_max - degree_min)*rand_factor
        degree_factor = degree_min * self.ones((bs, 1), mstype.float32)
        rand_factor = (check_rand * degree_factor) + (~check_rand * rand_factor)
        rand_factor = self.reshape(C.repeat_elements(rand_factor, rep=(h*w*c)), (bs, h, w, c))

        return rand_factor


class RandomColorAdjust(nn.Cell):
    """
    Applies Random Color Adjust transform on given input tensors.
    """

    def __init__(self, brightness, contrast, saturation, hue):
        super(RandomColorAdjust, self).__init__()

        if isinstance(brightness, (list, tuple)):
            self.br_min = brightness[0]
            self.br_max = brightness[1]
        else:
            self.br_min = max(0, 1 - brightness)
            self.br_max = 1 + brightness

        if isinstance(contrast, (list, tuple)):
            self.cont_min = contrast[0]
            self.cont_max = contrast[1]
        else:
            self.cont_min = max(0, 1 - contrast)
            self.cont_max = 1 + contrast

        if isinstance(saturation, (list, tuple)):
            self.sa_min = saturation[0]
            self.sa_max = saturation[1]
        else:
            self.sa_min = max(0, 1 - saturation)
            self.sa_max = 1 + saturation

        if isinstance(hue, (list, tuple)):
            self.hue_min = hue[0]
            self.hue_max = hue[1]
        else:
            self.hue_min = max(0, 1 - hue)
            self.hue_max = 1 + hue

        self.check_rand_br = Tensor(self.br_min == self.br_max)
        self.check_rand_cont = Tensor(self.cont_min == self.cont_max)
        self.check_rand_sa = Tensor(self.sa_min == self.sa_max)
        self.check_rand_hue = Tensor(self.hue_min == self.hue_max)

        self.cast = P.Cast()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.unstack = P.Unstack(axis=-1)
        self.expand_dims = P.ExpandDims()
        self.mul = P.Mul()

        self.mean = P.ReduceMean()
        self.argmaxvalue = P.ArgMaxWithValue(axis=3, keep_dims=False)
        self.argminvalue = P.ArgMinWithValue(axis=3, keep_dims=False)
        self.stack = P.Stack(axis=0)
        self.epsilon = Tensor(np.finfo(np.float32).eps, mstype.float32)
        self.squeeze_0 = P.Squeeze(axis=0)
        self.expand_dims = P.ExpandDims()
        self.gatherd = P.GatherD()

        self.generate_rand_batch = GenerateRandBatch()

    def construct(self, x):

        x = self.cast(x, mstype.float32)
        x_shape = self.shape(x)
        check_input_dims(x_shape, 4, 'RandomColorAdjust')
        bs, h, w, c = x_shape

        br_rand_factor = self.generate_rand_batch(self.br_min, self.br_max, self.check_rand_br, x_shape)
        cont_rand_factor = self.generate_rand_batch(self.cont_min, self.cont_max, self.check_rand_cont, x_shape)
        sat_rand_factor = self.generate_rand_batch(self.sa_min, self.sa_max, self.check_rand_sa, x_shape)

        r, g, b = self.unstack(x)

        x_gray = 0.2989 * r + 0.587 * g + 0.114 * b
        x_gray_mean = self.expand_dims(self.mean(x_gray, (1, 2)) + 0.5, -1)
        x_gray_mean = self.reshape(C.repeat_elements(x_gray_mean, rep=(h*w*c)), (bs, h, w, c))
        x_gray = C.repeat_elements(self.expand_dims(x_gray, -1), rep=c, axis=-1)

        # Apply brightness
        x = self.mul(x, br_rand_factor)
        x = C.clip_by_value(x, 0.0, 255.0)

        # Apply contrast
        x = self.mul(x, cont_rand_factor) + self.mul((1 - cont_rand_factor), x_gray_mean)
        x = C.clip_by_value(x, 0.0, 255.0)

        # Apply saturation
        x = self.mul(x, sat_rand_factor) + self.mul((1 - sat_rand_factor), x_gray)
        x = C.clip_by_value(x, 0.0, 255.0)

        # Apply Hue Transform
        # Convert tensor from rgb to hsv
        r, g, b = self.unstack(x)
        max_c, max_v = self.argmaxvalue(x)
        _, min_v = self.argminvalue(x)
        hsv_denum = max_v - min_v + self.epsilon
        h1 = self.floormod(((b - g) * 60 / hsv_denum), 360)
        h2 = (g - r) * 60 / hsv_denum + 120
        h3 = (r - g) * 60 / hsv_denum + 240
        h = self.stack((h1, h2, h3))
        h = self.gatherd(h, 0, self.expand_dims(max_c, 0))
        h = self.squeeze(h)
        s = self.cast((max_v > 0), mstype.float32)
        s = s * (1 - min_v / (max_v + self.epsilon))
        v = self.cast(max_v, mstype.float32)

        # Adjust hue
        hue_rand_factor = self.generate_rand_batch(self.hue_min, self.hue_max, self.check_rand_hue, x_shape)
        h = h + hue_rand_factor * 360.0

        # Convert tensor from hsv to rgb
        h_ = (h - self.floor(h / 360.0) * 360.0) / 60.0
        c = self.mul(s, v)
        x_ = self.mul(c, (1 - self.abs(self.fmod(h_, 2) - 1)))
        zero_tensor = self.zeros_like(c)

        y = self.stack((self.stack_axis_1((c, x_, zero_tensor)), self.stack_axis_1((x_, c, zero_tensor)),
                        self.stack_axis_1((zero_tensor, c, x_)), self.stack_axis_1((zero_tensor, x_, c)),
                        self.stack_axis_1((x_, zero_tensor, c)), self.stack_axis_1((c, zero_tensor, x_)),
                        ))

        index = self.expand_dims(self.floor(h_), 1)
        index = self.expand_dims(C.repeat_elements(index, 3, 1), 0)
        index = self.cast(index, mstype.int32)

        x = self.squeeze(self.gatherd(y, 0, index))
        x = x + self.reshape(C.repeat_elements((v - c), rep=(3)), self.shape(x))

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
        self.ones = P.Ones()
        self.reshape = P.Reshape()
        self.expand_dims = P.ExpandDims()
        self.mul = P.Mul()
        self.transpose = P.Transpose()

        self.check_rand = Tensor(self.degree_min == self.degree_max)

        self.weight = np.array([[1, 1, 1], [1, 5, 1], [1, 1, 1]])/13.0
        self.weight = np.repeat(self.weight[np.newaxis, :, :], 3, axis=0)
        self.weight = np.repeat(self.weight[np.newaxis, :, :], 3, axis=0)
        self.weight = Tensor(self.weight, mstype.float32)

        self.filter = P.Conv2D(out_channel=3, kernel_size=(3, 3), pad_mode='pad', pad=1)

    def construct(self, x):

        x = self.cast(x, mstype.float32)
        x_shape = self.shape(x)
        check_input_dims(x_shape, 4, 'RandomSharpness')
        bs, h, w, c = x_shape

        degree_rand_factor = Tensor(np.random.uniform(size=(bs, 1)), dtype=mstype.float32)
        degree_rand_factor = self.degree_min + (self.degree_max - self.degree_min)*degree_rand_factor
        degree_factor = self.degree_min * self.ones((bs, 1), mstype.float32)
        degree_rand_factor = (self.check_rand * degree_factor) + (~self.check_rand * degree_rand_factor)
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
        self.shape = P.Shape()

    def construct(self, x):
        x_shape = self.shape(x)
        check_input_dims(x_shape, 4, 'HwcToChw')
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


class TypeCast(nn.Cell):
    """
    Applies TypeCast transform on given input tensors.
    """

    def __init__(self, data_type_str):
        super(TypeCast, self).__init__()

        self.cast = P.Cast()
        self.data_type = mstype.typing.str_to_type(data_type_str)

    def construct(self, x):

        return self.cast(x, self.data_type)


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
    "Rescale": OffloadModel(Rescale, ["rescale", "shift"]),
    "TypeCast": OffloadModel(TypeCast, ["data_type"])
}


class GetModelFromJson2Col(nn.Cell):
    """
    Generates offload ME model from offload JSON file for a single map op.
    """

    def __init__(self, json_offload, col_idxs):
        super(GetModelFromJson2Col, self).__init__()
        self.col_idxs = col_idxs
        self.me_ops = []
        self.input_cols = []
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
        # apply single column
        col_idx = self.col_idxs[0]
        x[col_idx] = self.cell(x[col_idx])

        return x


class GetOffloadModel(nn.Cell):
    """
    Generates offload ME model.
    """

    def __init__(self, dataset_consumer, ds_cols):
        super(GetOffloadModel, self).__init__()
        self.transform_list = []
        json_offload = json.loads(dataset_consumer.GetOffload())
        if json_offload is not None:
            for node in json_offload:
                if node["op_type"] == 'Map':
                    ds_col_idxs = get_col_idxs(node["input_columns"], ds_cols)
                    self.transform_list.append(GetModelFromJson2Col(node, ds_col_idxs))
            self.transform_list.reverse()

    def construct(self, x):
        for transform in self.transform_list:
            x = transform(x)
        return x
