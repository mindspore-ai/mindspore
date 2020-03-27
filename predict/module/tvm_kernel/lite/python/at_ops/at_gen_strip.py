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
This module is rule to generation tvm operate. you can use it like:
python3 at_gen_strip.py [x86:arm64:arm32]
"""
import os
import sys
import itertools
from functools import partial
from at_ops.at_lib import Deconv, tvm, ConvVar, BatchNorm, Eltwise, Resize, CaffeCrop, CaffePReLU
from at_ops.at_lib import FullConnection, Power, ArgMax, Concat, Pad, Pooling, Mean, MatMul, Softmax
from at_ops.at_lib import Activation, Exp, Split, Cast, ExpandDims, Tile, Range
from at_rt import at_runtime_reset


check_correctness = False
ARCH_TYPE = sys.argv[1]

dtypes = ("float32",)  # "float16",  "uint8", "int8", "uint32", "int32"

device_map = {
    "x86": "llvm",
    "arm64": "llvm -device=arm_cpu -model=kirin970 -target=arm64-linux-android",
    "arm32": "llvm -device=arm_cpu -model=kirin970 -target=armv7a-linux-eabi -mfloat-abi=soft",
}

lib_path_map = {
    "x86": "../../../build/lib_x86/",
    "arm64": "../../../build/lib_arm64/",
    "arm32": "../../../build/lib_arm32/",
}

best_log_map = {
    "x86": None,
    "arm64": None,
    "arm32": None,
}

lib_path = lib_path_map[ARCH_TYPE]
device = device_map[ARCH_TYPE]
if ARCH_TYPE == "arm64":
    if dtypes[0] == "float16":
        device += " -mattr=+fp16fml"
    else:
        device += " -mattr=+neon"
best_log = best_log_map[ARCH_TYPE]

kwargs = {
    "device": device,
    "lib_path": lib_path,
    "check_correctness": check_correctness,
}

use_arm32 = ARCH_TYPE == "arm32"

MAX_DIMS = 5
const_op_list = [
    (
        "Deconvolution",
        partial(Deconv, optype="Deconvolution"),
        {
            "ndim": (5,),
            "dtype": dtypes,
            "kernels": ((2, 2),),
            "strides": ((2, 2),),
            "pad": ((0, 0, 0, 0),),
            "dilations": ((1, 1),),
            "hasbias": (False, True),
            "activation_type": ("NO_ACTIVATION",),
            "cfg": [
                {
                    "CI": tvm.var("CI"),
                    "VH": 2,
                    "VW": 12,
                    "VC": 4,
                    "VI": 4,
                    "tile_oh": 2,
                    "tile_ow": 12,
                    "tile_co": 4,
                    "ann_reduce": ["none", "unroll"],
                    "ann_spatial": ["unroll", "unroll", "vec"],
                },
                {
                    "CI": tvm.var("CI"),
                    "VH": 2,
                    "VW": 10,
                    "VC": 4,
                    "VI": 4,
                    "tile_oh": 2,
                    "tile_ow": 10,
                    "tile_co": 4,
                    "ann_reduce": ["none", "unroll"],
                    "ann_spatial": ["unroll", "unroll", "vec"],
                },
                {
                    "CI": tvm.var("CI"),
                    "VH": 2,
                    "VW": 16,
                    "VC": 4,
                    "VI": 4,
                    "tile_oh": 2,
                    "tile_ow": 16,
                    "tile_co": 4,
                    "ann_reduce": ["none", "unroll"],
                    "ann_spatial": ["unroll", "unroll", "vec"],
                },
                {
                    "CI": tvm.var("CI"),
                    "VH": 2,
                    "VW": 8,
                    "VC": 4,
                    "VI": 4,
                    "tile_oh": 2,
                    "tile_ow": 8,
                    "tile_co": 4,
                    "ann_reduce": ["none", "unroll"],
                    "ann_spatial": ["unroll", "unroll", "vec"],
                },
                {
                    "CI": tvm.var("CI"),
                    "VH": 2,
                    "VW": 4,
                    "VC": 4,
                    "VI": 4,
                    "tile_oh": 2,
                    "tile_ow": 4,
                    "tile_co": 4,
                    "ann_reduce": ["none", "unroll"],
                    "ann_spatial": ["unroll", "unroll", "vec"],
                },
                {
                    "CI": tvm.var("CI"),
                    "VH": 2,
                    "VW": 2,
                    "VC": 4,
                    "VI": 4,
                    "tile_oh": 2,
                    "tile_ow": 2,
                    "tile_co": 4,
                    "ann_reduce": ["none", "unroll"],
                    "ann_spatial": ["unroll", "unroll", "vec"],
                },
            ],
        },
    ),
    (
        "Convolution",
        partial(ConvVar, optype="Convolution"),
        {
            "ndim": (4,),
            "layout": ("NCHW",),
            "dtype": dtypes,
            "kernels": ((1, 1), (3, 3), (5, 5),),
            "strides": ((1, 1), (2, 2)),
            "pad": ((1, 1, 1, 1), (0, 0, 0, 0), (2, 2, 2, 2)),
            "dilations": ((1, 1),),
            "hasbias": (False, True),
            "activation_type": ("NO_ACTIVATION", "RELU"),
            "cfg": [
                {
                    "CI": tvm.var("CI"),
                    "VH": 1,
                    "VW": 1,
                    "VC": 1,
                    "VI": 1,
                    "tile_oh": 1,
                    "tile_ow": 1,
                    "tile_co": 1,
                    "ann_reduce": ["none", "unroll"],
                    "ann_spatial": ["unroll", "unroll", "vec"],
                    "core_id": 0,
                },
            ],
        },
    ),
    (
        "ConvolutionDepthwise",
        partial(ConvVar, optype="ConvolutionDepthwise"),
        {
            "ndim": (4,),
            "layout": ("NCHW",),
            "dtype": dtypes,
            "kernels": ((2, 2), (3, 3),),
            "strides": ((1, 1),),
            "pad": ((0, 0, 0, 0), (0, 1, 0, 1), (1, 0, 1, 0), (1, 1, 1, 1),),
            "dilations": ((1, 1),),
            "hasbias": (False, True),
            "activation_type": ("NO_ACTIVATION", "RELU"),
            "channel_multiplier": (1,),
            "cfg": [
                {
                    "CI": tvm.var("CI"),
                    "VH": 1,
                    "VW": 1,
                    "VC": 1,
                    "VI": 1,
                    "tile_oh": 1,
                    "tile_ow": 1,
                    "tile_co": 1,
                    "ann_reduce": ["none", "unroll"],
                    "ann_spatial": ["unroll", "unroll", "vec"],
                    "core_id": 0,
                },
            ],
        },
    ),
    (
        "DeConvolutionDepthwise",
        partial(ConvVar, optype="DeConvolutionDepthwise"),
        {
            "ndim": (4,),
            "layout": ("NCHW",),
            "dtype": dtypes,
            "kernels": ((1, 1), (2, 2), (3, 3),),
            "strides": ((1, 1), (2, 2),),
            "pad": ((0, 0, 0, 0), (1, 0, 1, 0), (1, 1, 1, 1),),
            "dilations": ((1, 1),),
            "hasbias": (False, True),
            "activation_type": ("NO_ACTIVATION", "RELU"),
            "channel_multiplier": (1,),
            "cfg": [
                {
                    "CI": tvm.var("CI"),
                    "VH": 1,
                    "VW": 1,
                    "VC": 1,
                    "VI": 1,
                    "tile_oh": 1,
                    "tile_ow": 1,
                    "tile_co": 1,
                    "ann_reduce": ["none", "unroll"],
                    "ann_spatial": ["unroll", "unroll", "vec"],
                    "core_id": 0,
                },
            ],
        },
    ),
    (
        "BatchNorm",
        BatchNorm,
        {"ndim": (4,), "dtype": dtypes, "optype": ("TFBatchNorm",), "axis": (1, 3,)},
    ),
    (
        "BiasAdd",
        BatchNorm,
        {"ndim": (2, 4), "dtype": dtypes, "optype": ("TFBiasAdd",), "axis": (1, 3)},
    ),
    (
        "CaffeBatchNorm",
        BatchNorm,
        {"ndim": (2, 4), "dtype": dtypes, "optype": ("CaffeBatchNorm",), "axis": (1, 3)},
    ),
    (
        "Scale",
        BatchNorm,
        {"ndim": (2, 4), "dtype": dtypes, "optype": ("CaffeScale",), "axis": (1,)},
    ),
    (
        "Eltwise",
        Eltwise,
        {
            "ndim_a": tuple(range(0, MAX_DIMS + 1)),
            "ndim_b": tuple(range(0, MAX_DIMS + 1)),
            "dtype": dtypes,
            "mode": ("add", "subtract", "multiply", "divide", "maximum"),
        },
    ),
    (
        "Add",
        Eltwise,
        {
            "ndim_a": tuple(range(0, MAX_DIMS + 1)),
            "ndim_b": tuple(range(0, MAX_DIMS + 1)),
            "dtype": dtypes,
            "mode": ("add",),
        },
    ),
    (
        "Sub",
        Eltwise,
        {
            "ndim_a": tuple(range(0, MAX_DIMS + 1)),
            "ndim_b": tuple(range(0, MAX_DIMS + 1)),
            "dtype": dtypes,
            "mode": ("subtract",),
        },
    ),
    (
        "Mul",
        Eltwise,
        {
            "ndim_a": tuple(range(0, MAX_DIMS + 1)),
            "ndim_b": tuple(range(0, MAX_DIMS + 1)),
            "dtype": dtypes,
            "mode": ("multiply",),
        },
    ),
    (
        "RealDiv",
        Eltwise,
        {
            "ndim_a": tuple(range(0, MAX_DIMS + 1)),
            "ndim_b": tuple(range(0, MAX_DIMS + 1)),
            "dtype": dtypes,
            "mode": ("divide",),
        },
    ),
    (
        "Maximum",
        Eltwise,
        {
            "ndim_a": tuple(range(0, MAX_DIMS + 1)),
            "ndim_b": tuple(range(0, MAX_DIMS + 1)),
            "dtype": dtypes,
            "mode": ("maximum",),
        },
    ),
    (
        "ResizeBilinear",
        Resize,
        {
            "ndim": (4,),
            "dtype": dtypes,
            "method": ("bilinear",),  # "bicubic"
            "align_corners": (True, False),
        },
    ),
    (
        "ResizeNearestNeighbor",
        Resize,
        {
            "ndim": (4,),
            "dtype": dtypes,
            "method": ("nearest_neighbor",),  # "bicubic"
            "align_corners": (True, False),
        },
    ),
    (
        "CaffeCrop",
        CaffeCrop,
        {"ndim": (4,), "dtype": dtypes, "axis": tuple(range(0, 4))},
    ),
    (
        "CaffePReLU",
        CaffePReLU,
        {"ndim": (2, 4), "dtype": dtypes, "channel_shared": (True, False)},
    ),
    (
        "FullConnection",
        FullConnection,
        {"ndim_a": (2, 4), "dtype": dtypes, "has_bias": (True, False)},
    ),
    ("Power", Power, {"ndim": tuple(range(1, MAX_DIMS + 1)), "dtype": dtypes}),
    (
        "ArgMax",
        ArgMax,
        {
            "ndim": tuple(range(1, MAX_DIMS + 1)),
            "dtype": dtypes,
            "axis": tuple(range(0, MAX_DIMS)),  # not support None
            "keep_dims": (True, False),
            "top_k": (1,),
            "out_dtype": dtypes,
        },
    ),
    (
        "Concat",
        Concat,
        {
            "ndim": tuple(range(1, MAX_DIMS + 1)),
            "dtype": dtypes,
            "input_num": tuple(range(2, 6 + 1)),
            "axis": tuple(range(0, MAX_DIMS)),
        },
    ),
    (
        "Pad",
        Pad,
        {
            "ndim": tuple(range(2, MAX_DIMS + 1)),
            "dtype": dtypes,
            "paddingmode": ("CONSTANT", "REFLECT", "SYMMETRIC"),
        },
    ),
    (
        "Pooling",
        Pooling,
        {
            "ndim": (4,),
            "dtype": dtypes,
            "pooling_mode": ("max", "avg"),
            "caffe_mode": (True, False),
            "kernel": ((1, 1), (2, 2), (3, 3), (5, 5)),
            "stride": ((1, 1), (2, 2), (3, 3)),
            "pad": ((0, 0, 0, 0), (0, 1, 0, 1), (1, 1, 1, 1)),
            "use_global": (True, False),
        },
    ),
    (
        "Mean",
        Mean,
        {
            "ndim": (4,),
            "dtype": dtypes,
            "axis": (
                (0,),
                (1,),
                (2,),
                (3,),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3),
                (2, 3),
                (0, 1, 2),
                (0, 1, 3),
                (0, 2, 3),
                (1, 2, 3),
                (0, 1, 2, 3),
            ),
            "keep_dims": (True, False),
        },
    ),
    (
        "MatMul",
        MatMul,
        {
            "ndim_a": (2,),
            "ndim_b": (2,),
            "dtype": dtypes,
            "transpose_a": (True, False),
            "transpose_b": (True, False),
        },
    ),
    (
        "Softmax",
        Softmax,
        {
            "ndim": tuple(range(1, MAX_DIMS + 1)),
            "dtype": dtypes,
            "axis": tuple(range(0, MAX_DIMS)),
        },
    ),
    (
        "Activation",
        Activation,
        {
            "ndim": tuple(range(1, MAX_DIMS + 1)),
            "dtype": dtypes,
            "optype": ("NO_ACTIVATION", "RELU", "RELU6", "SIGMOID"),
        },
    ),
    ("Exp", Exp, {"ndim": tuple(range(1, MAX_DIMS + 1)), "dtype": dtypes}),
    (
        "Split",
        Split,
        {
            "ndim": tuple(range(1, MAX_DIMS + 1)),
            "dtype": dtypes,
            "output_num": tuple(range(1, 5)),
            "axis": tuple(range(0, MAX_DIMS)),
        },
    ),
    (
        "Cast",
        Cast,
        {
            "ndim": tuple(range(1, MAX_DIMS + 1)),
            "src_dtype": dtypes,
            "dst_dtype": dtypes,
        },
    ),
    (
        "ExpandDims",
        ExpandDims,
        {
            "ndim": tuple(range(1, MAX_DIMS + 1)),
            "dtype": dtypes,
            "axis": tuple(range(0, MAX_DIMS)),
        },
    ),
    ("Tile", Tile, {"ndim": tuple(range(1, MAX_DIMS + 1)), "dtype": dtypes}),
    ("Range", Range, {"out_dtype": ("float32", "uint32", "int32")}),
]


def gen_const_libs(some_op=None):
    for optype, func, attr in const_op_list:
        if some_op and some_op != optype:
            continue
        for values in itertools.product(*attr.values()):
            args = dict((k, v) for k, v in zip(attr.keys(), values))
            func(device=device, lib_path=lib_path, **args)


if __name__ == "__main__":
    if not os.path.exists(lib_path):
        os.makedirs(lib_path)
    # skip best_history log:
    with tvm.target.create(device):
        with at_runtime_reset.AtRuntimeReset():
            gen_const_libs()
