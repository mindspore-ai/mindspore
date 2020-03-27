# Copyright 2019 Huawei Technologies Co., Ltd
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
"""
This module is define some data struct for tvm kernel.
"""
import tvm
import topi

format_map = {"NCHW": 0, "NHWC": 1}

pool_map = {"max_pool": 0, "avg_pool": 1, "global_pool": 2}

activation_map = {
    "no_activation": 0,
    "relu": 1,
    "sigmoid": 2,
    "relu6": 3,
    "elu": 4,
    "leaky_relu": 5,
    "abs": 6,
    "relu1": 7,
    "softsign": 8,
    "softplus": 9,
    "tanh ": 10,
}
activation_enum_map = {
    "NO_ACTIVATION": 0,
    "RELU": 1,
    "SIGMOID": 2,
    "RELU6": 3,
    "elu": 4,
    "leaky_relu": 5,
    "abs": 6,
    "relu1": 7,
    "softsign": 8,
    "softplus": 9,
    "tanh ": 10,
}

padmode_map = {"NOTSET": 0, "SAME": 1, "VALID": 2}

mslite_datatype_map = {
    "float16": 1,
    "float32": 0,
    "double": 11,
    "int8": 2,
    "int16": 6,
    "int32": 3,
    "int64": 9,
    "uint8": 4,
    "uint16": 7,
    "uint32": 8,
    "uint64": 10,
}


def get_key_by_value(dicts, value):
    for k, v in dicts.items():
        if v == value:
            return k
    return None


def relu6(x):
    return tvm.compute(
        x.shape,
        lambda *i: tvm.min(
            tvm.max(x(*i), tvm.const(0, x.dtype)), tvm.const(6, x.dtype)
        ),
    )


activation_topi_funs = {"NO_ACTIVATION": None, "RELU": topi.nn.relu, "RELU6": relu6}

name_funcs = {
    "Concat": (
        lambda opname, x: (
            opname + "_%d_%d" + "_%d" + "_%d" * x["ndim"] + "_%d" * len(x["shapeAxis"])
        )
        % (
            format_map[x["format"]],
            x["ndim"],
            x["axis"],
            *x["shapeOut"],
            *x["shapeAxis"],
        )
    ),
    "Softmax": (
        lambda opname, x: (opname + "_%d_%d" + "_%d" * x["ndim"] + "_%d")
        % (format_map[x["format"]], x["ndim"], *x["shape"], x["axis"])
    ),
    "Activation": (
        lambda opname, x: (opname + "_%d_%d" + "_%d" + "_%d" * x["ndim"])
        % (format_map[x["format"]], x["ndim"], activation_map[x["type"]], *x["shape"])
    ),
    "Add": (
        lambda opname, x: (opname + "_%d_%d" + "_%d" * x["ndim"])
        % (format_map[x["format"]], x["ndim"], *x["shape"])
    ),
    "Convolution": (
        lambda opname, x: (
            opname + "_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d"
        )
        % (
            format_map[x["format"]],
            x["ndim"],
            x["batch"],
            x["in_channel"],
            *x["in_size"],
            x["num_filter"],
            *x["filter_size"],
            *x["pad"],
            *x["stride"],
            x["dilation"],
            x["hasbias"],
            activation_map[x["activation_type"]],
        )
    ),
    "Identity": (
        lambda opname, x: (opname + "_%d_%d" + "_%d" * x["ndim"])
        % (format_map[x["format"]], x["ndim"], *x["shape"])
    ),
    "BatchNorm": (
        lambda opname, x: (opname + "_%d_%d" + "_%d" * x["ndim"] + "_%d")
        % (format_map[x["format"]], x["ndim"], *x["shape"], x["epsilon"])
    ),
    "Squeeze": (
        lambda opname, x: (
            opname + "_%d_%d" + "_%d" * x["ndim"] + "_%d" * len(x["axis"])
        )
        % (format_map[x["format"]], x["ndim"], *x["shape"], *x["axis"])
    ),
    "BiasAdd": (
        lambda opname, x: (opname + "_%d_%d" + "_%d" * x["ndim"] + "_%d")
        % (format_map[x["format"]], x["ndim"], *x["shape"], x["axis"])
    ),
    "Pooling": (
        lambda opname, x: (opname + "_%d_%d_%d" + "_%d" * x["ndim"] + "_%d_%d_%d")
        % (
            format_map[x["format"]],
            x["ndim"],
            pool_map[x["type"]],
            *x["shape"],
            x["kernel"],
            x["stride"],
            x["pad"],
        )
    ),
    "ConvolutionDepthwise": (
        lambda opname, x: (
            opname + "_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d"
        )
        % (
            format_map[x["format"]],
            x["ndim"],
            x["batch"],
            x["in_channel"],
            *x["in_size"],
            x["in_channel"] * x["channel_multiplier"],
            *x["filter_size"],
            *x["pad"],
            *x["stride"],
            x["dilation"],
            x["hasbias"],
            activation_map[x["activation_type"]],
        )
    ),
    "Reshape": (
        lambda opname, x: (
            opname + "_%d_%d" + "_%d" * x["ndimA"] + "_%d" * len(x["shapeB"])
        )
        % (format_map[x["format"]], x["ndimA"], *x["shapeA"], *x["shapeB"])
    ),
    "Shape": (
        lambda opname, x: (opname + "_%d_%d" + "_%d" * x["ndim"])
        % (format_map[x["format"]], x["ndim"], *x["shape"])
    ),
    "RealDiv": (
        lambda opname, x: (
            opname + "_%d_%d" + "_%d" * x["ndim"] + "_%d" * len(x["shapeB"])
        )
        % (format_map[x["format"]], x["ndim"], *x["shapeA"], *x["shapeB"])
    ),
    "ResizeBilinear": (lambda opname, x: "ResizeBilinear"),
    "TFLite_Detection_PostProcess": (lambda opname, x: "TFLite_Detection_PostProcess"),
}

config_dict = {op_type: [] for op_type in name_funcs}


def config_dict_append(op_type, config, opname=None):
    if opname is None:
        config["opname"] = name_funcs[op_type](op_type, config)
    else:
        config["opname"] = opname
    duplicate = [True for x in config_dict[op_type] if config == x]

    if duplicate:
        config_dict[op_type].append(config)
