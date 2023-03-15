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
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.ops.composite as C
from mindspore.ops import operations as P
from mindspore import log as logger


def check_add_offload_sink_mode(dataset, dataset_helper, network):
    """
    Check if any map operations were removed to be offloaded and apply the transforms if so.
    """
    if hasattr(dataset, '__no_send__'):
        # Dataset was not sent to device. Skip adding offload.
        return network
    offload_model = dataset.__transfer_dataset__.get_offload_model()
    # See if the offload pass identified any operations to be offloaded
    if offload_model.transform_list != []:
        check_concat_zip_dataset(dataset.__transfer_dataset__)
        network = ApplyPreTransform(offload_model, network)
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

    data = offload_model(*data)
    data = list(data)
    for idx in non_tensor_idxs:
        data[idx] = data[idx].asnumpy()

    return data


def assign_min_max_params(in_params, center=1):
    """
    Adjust input parameters for ops.
    """
    if isinstance(in_params, (list, tuple)):
        min_param = in_params[0]
        max_param = in_params[1]
    else:
        min_param = max(0, center - in_params)
        max_param = center + in_params

    return min_param, max_param


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

        data = self.transform(*data)
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
        bs, h, w, c = x_shape

        flip_rand_factor = Tensor(np.random.uniform(size=(bs, 1)), dtype=mstype.float32)
        flip_rand_factor = self.cast((self.prob > flip_rand_factor), mstype.float32)
        flip_rand_factor = self.reshape(C.repeat_elements(flip_rand_factor, rep=(h*w*c)), (bs, h, w, c))

        x_flip = self.h_flip(x)
        operation = self.mul(x_flip, flip_rand_factor) + self.mul((1 - flip_rand_factor), x)
        # ops.depend is added to find the RandomHorizontalFlip operator in IR files.
        depend = ops.depend(operation, "RandomHorizontalFlip")
        return depend


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
        bs, h, w, c = x_shape

        flip_rand_factor = Tensor(np.random.uniform(size=(bs, 1)), dtype=mstype.float32)
        flip_rand_factor = self.cast((self.prob > flip_rand_factor), mstype.float32)
        flip_rand_factor = self.reshape(C.repeat_elements(flip_rand_factor, rep=(h*w*c)), (bs, h, w, c))

        x_flip = self.h_flip(x)
        operation = self.mul(x_flip, flip_rand_factor) + self.mul((1 - flip_rand_factor), x)
        # ops.depend is added to find the RandomVerticalFlip operator in IR files.
        depend = ops.depend(operation, "RandomVerticalFlip")
        return depend


class GenerateRandBatch(nn.Cell):
    """
    Generate batch with random values uniformly selected from [degree_min, degree_max].
    """

    def __init__(self):
        super(GenerateRandBatch, self).__init__()

        self.ones = P.Ones()
        self.reshape = P.Reshape()

    def construct(self, degree_min, degree_max, check_rand, shape):

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

        self.br_min, self.br_max = assign_min_max_params(brightness)
        self.cont_min, self.cont_max = assign_min_max_params(contrast)
        self.sa_min, self.sa_max = assign_min_max_params(saturation)
        self.hue_min, self.hue_max = assign_min_max_params(hue)

        self.check_rand_br = Tensor(self.br_min == self.br_max)
        self.check_rand_cont = Tensor(self.cont_min == self.cont_max)
        self.check_rand_sa = Tensor(self.sa_min == self.sa_max)
        self.check_rand_hue = Tensor(self.hue_min == self.hue_max)

        self.cast = P.Cast()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.unstack = P.Unstack(axis=-1)
        self.unstack_dim_1 = P.Unstack(axis=1)
        self.expand_dims = P.ExpandDims()
        self.mul = P.Mul()

        self.mean = P.ReduceMean()
        self.argmaxvalue = P.ArgMaxWithValue(axis=1, keep_dims=False)
        self.argminvalue = P.ArgMinWithValue(axis=1, keep_dims=False)
        self.stack = P.Stack(axis=0)
        self.epsilon = Tensor(np.finfo(np.float32).eps, mstype.float32)
        self.squeeze = P.Squeeze(axis=0)
        self.expand_dims = P.ExpandDims()
        self.gatherd = P.GatherD()
        self.floor = P.Floor()
        self.fmod = P.FloorMod()
        self.abs = P.Abs()
        self.zeros_like = P.ZerosLike()
        self.stack_axis_1 = P.Stack(axis=1)
        self.transpose = P.Transpose()
        self.ones = P.Ones()
        self.reshape = P.Reshape()

        self.generate_rand_batch = GenerateRandBatch()

    def construct(self, x):

        x = self.cast(x, mstype.float32)
        x_shape = self.shape(x)
        bs, h, w, c = x_shape

        br_rand_factor = self.generate_rand_batch(self.br_min, self.br_max, self.check_rand_br, x_shape)
        cont_rand_factor = self.generate_rand_batch(self.cont_min, self.cont_max, self.check_rand_cont, x_shape)
        sat_rand_factor = self.generate_rand_batch(self.sa_min, self.sa_max, self.check_rand_sa, x_shape)

        r_, g_, b_ = self.unstack(x)

        x_gray = 0.2989 * r_ + 0.587 * g_ + 0.114 * b_
        x_gray_mean = self.expand_dims(self.mean(x_gray, (1, 2)) + 0.5, -1)
        x_gray_mean = self.reshape(C.repeat_elements(x_gray_mean, rep=(h*w*c)), (bs, h, w, c))
        x_gray = C.repeat_elements(self.expand_dims(x_gray, -1), rep=c, axis=-1)

        # Apply brightness
        x = self.mul(x, br_rand_factor)
        x = ops.clip_by_value(x, 0.0, 255.0)

        # Apply contrast
        x = self.mul(x, cont_rand_factor) + self.mul((1 - cont_rand_factor), x_gray_mean)
        x = ops.clip_by_value(x, 0.0, 255.0)

        # Apply saturation
        x = self.mul(x, sat_rand_factor) + self.mul((1 - sat_rand_factor), x_gray)
        x = ops.clip_by_value(x, 0.0, 255.0)

        # Apply Hue Transform
        # Convert tensor from rgb to hsv
        x_swap = self.transpose(x, (0, 3, 1, 2)) / 255.0
        r, g, b = self.unstack_dim_1(x_swap)
        _, max_rgb = self.argmaxvalue(x_swap)
        argmin_rgb, min_rgb = self.argminvalue(x_swap)

        max_min = max_rgb - min_rgb + self.epsilon
        h1 = (g - r) * 60 / max_min + 60
        h2 = (b - g) * 60 / max_min + 180
        h3 = (r - b) * 60 / max_min + 300
        hue = self.squeeze(self.gatherd(self.stack((h2, h3, h1)), 0, self.expand_dims(argmin_rgb, 0)))
        s = max_min / (max_rgb + self.epsilon)
        v = max_rgb

        # Adjust hue
        hue_rand_factor = Tensor(np.random.uniform(size=(bs, 1)), dtype=mstype.float32)
        hue_rand_factor = self.hue_min + (self.hue_max - self.hue_min)*hue_rand_factor
        degree_factor = self.hue_min * self.ones((bs, 1), mstype.float32)
        hue_rand_factor = (self.check_rand_hue * degree_factor) + (~self.check_rand_hue * hue_rand_factor)
        hue_rand_factor = self.reshape(C.repeat_elements(hue_rand_factor, rep=(h*w)), (bs, h, w))
        hue = hue + (hue_rand_factor * 360.0)

        # Convert tensor from hsv to rgb
        h_ = (hue - self.floor(hue / 360.0) * 360.0) / 60.0
        c = s * v
        x_ = c * (1 - self.abs(self.fmod(h_, 2) - 1))

        zero_tensor = self.zeros_like(c)
        y = self.stack((self.stack_axis_1((c, x_, zero_tensor)),
                        self.stack_axis_1((x_, c, zero_tensor)),
                        self.stack_axis_1((zero_tensor, c, x_)),
                        self.stack_axis_1((zero_tensor, x_, c)),
                        self.stack_axis_1((x_, zero_tensor, c)),
                        self.stack_axis_1((c, zero_tensor, x_)),
                        ))

        index = self.expand_dims(self.floor(h_), 1)
        index = self.cast(self.expand_dims(C.repeat_elements(index, 3, 1), 0), mstype.int32)
        x_rgb = self.squeeze(self.gatherd(y, 0, index))
        x_rgb = x_rgb + C.repeat_elements(self.expand_dims((v - c), 1), 3, 1)

        x_rgb = self.transpose(x, (0, 2, 3, 1)) * 255.0
        operation = ops.clip_by_value(x, 0.0, 255.0)
        # ops.depend is added to find the RandomColorAdjust operator in IR files.
        depend = ops.depend(operation, "RandomColorAdjust")
        return depend


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
        bs, h, w, c = x_shape

        degree_rand_factor = Tensor(np.random.uniform(size=(bs, 1)), dtype=mstype.float32)
        degree_rand_factor = self.degree_min + (self.degree_max - self.degree_min)*degree_rand_factor
        degree_factor = self.degree_min * self.ones((bs, 1), mstype.float32)
        degree_rand_factor = (self.check_rand * degree_factor) + (~self.check_rand * degree_rand_factor)
        degree_rand_factor = self.reshape(C.repeat_elements(degree_rand_factor, rep=(h*w*c)), (bs, h, w, c))

        x_sharp = self.filter(self.transpose(x, (0, 3, 1, 2)), self.weight)
        x_sharp = self.transpose(x_sharp, (0, 2, 3, 1))

        x = self.mul(x, degree_rand_factor) + self.mul((1 - degree_rand_factor), x_sharp)
        operation = ops.clip_by_value(x, 0.0, 255.0)
        # ops.depend is added to find the RandomSharpness operator in IR files.
        depend = ops.depend(operation, "RandomSharpness")
        return depend


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
        operation = x * self.rescale + self.shift
        # ops.depend is added to find the Rescale operator in IR files.
        depend = ops.depend(operation, "Rescale")
        return depend


class HwcToChw(nn.Cell):
    """
    Applies Channel Swap transform on given input tensors.
    """

    def __init__(self):
        super(HwcToChw, self).__init__()
        self.trans = P.Transpose()

    def construct(self, x):
        operation = self.trans(x, (0, 3, 1, 2))
        # ops.depend is added to find the HwcToChw operator in IR files.
        depend = ops.depend(operation, "HwcToChw")
        return depend


class Normalize(nn.Cell):
    """
    Applies Normalize transform on given input tensors.
    """

    def __init__(self, mean, std, is_hwc=True):
        super(Normalize, self).__init__()
        if is_hwc is False:
            mean = np.expand_dims(np.array(mean), (1, 2))
            std = np.expand_dims(np.array(std), (1, 2))
        self.mean = Tensor(mean, mstype.float32)
        self.std = Tensor(std, mstype.float32)
        self.sub = P.Sub()
        self.div = P.Div()
        self.cast = P.Cast()

    def construct(self, x):
        x = self.cast(x, mstype.float32)
        x = self.sub(x, self.mean)
        operation = self.div(x, self.std)
        # ops.depend is added to find the Normalize operator in IR files.
        depend = ops.depend(operation, "Normalize")
        return depend


class TypeCast(nn.Cell):
    """
    Applies TypeCast transform on given input tensors.
    """

    def __init__(self, data_type_str):
        super(TypeCast, self).__init__()

        self.cast = P.Cast()
        self.data_type = mstype.typing.str_to_type(data_type_str)

    def construct(self, x):
        operation = self.cast(x, self.data_type)
        # ops.depend is added to find the TypeCast operator in IR files.
        depend = ops.depend(operation, "TypeCast")
        return depend


class OffloadModel():
    def __init__(self, func, args_names=None):
        self.func = func
        self.args_names = args_names


# Dictionary connecting operation name to model
op_to_model = {
    "HWC2CHW": OffloadModel(HwcToChw),
    "HwcToChw": OffloadModel(HwcToChw),
    "Normalize": OffloadModel(Normalize, ["mean", "std", "is_hwc"]),
    "RandomColorAdjust": OffloadModel(RandomColorAdjust, ["brightness", "contrast", "saturation", "hue"]),
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

        # Check if input_culmns attr is empty in Map op.
        if not self.col_idxs:
            self.col_idxs = [0]
            logger.warning(
                "input_columns attr in map op is not defined, "
                "so offload op will be applied to first column of dataset.")

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

    def construct(self, *x):
        data = []
        for d in x:
            data.append(d)

        for transform in self.transform_list:
            data = transform(data)
        return data
