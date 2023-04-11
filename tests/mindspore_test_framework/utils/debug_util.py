# Copyright 2020 Huawei Technologies Co., Ltd
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

"""Debugging utils."""

# pylint: disable=missing-docstring, unused-argument

import logging

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops._grad_experimental.grad_base import bprop_getters
from mindspore.ops.primitive import prim_attr_register, PrimitiveWithInfer

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(asctime)s %(pathname)s:%(lineno)d %(message)s')
logger = logging.getLogger(__name__)


class PrintShapeType(PrimitiveWithInfer):
    """
    PrintShapeType input's shape and type.

    Args:
        msg (str): The msg to print.

    Inputs:
        - **input_x** (:class:`mindspore.dtype`) - The data to print.

    Outputs:
        Tensor, return x directly, PrintShapeType does not affect the forward and gradient result.

    Examples:
        >>> class PrintShapeType(nn.Cell):
        >>>    def __init__(self):
        >>>        super(PrintShapeType, self).__init__()
        >>>    def construct(self, msg, x):
        >>>        P.PrintShapeType(msg)(x)
        >>>        return x
        >>>
        >>> class PrintShapeTypeGrad(nn.Cell):
        >>>    def __init__(self, msg):
        >>>        super(PrintShapeTypeGrad, self).__init__()
        >>>        self.print_shape_type = P.InsertGradientOf(P.PrintShapeType(msg))
        >>>    def construct(self, x):
        >>>        self.print_shape_type(x)
        >>>        return x
    """

    @prim_attr_register
    def __init__(self, msg):
        super(PrintShapeType, self).__init__('PrintShapeType')
        self.msg = msg

    def __call__(self, x):
        logger.info('%s, data: %s', self.msg, x)
        return x

    def infer_shape(self, x_shape):
        logger.info('%s, shape: %s', self.msg, x_shape)
        return x_shape

    def infer_dtype(self, x_type):
        logger.info('%s, type: %s', self.msg, x_type)
        return x_type


@bprop_getters.register(PrintShapeType)
def get_bprop_print_shape_type(self):
    """Generate bprop for PrintShapeType"""

    def bprop(x, out, dout):
        return (dout,)

    return bprop


class PrintShapeTypeCell(nn.Cell):
    def construct(self, msg, x):
        PrintShapeType(msg)(x)
        return x


class PrintGradShapeTypeCell(nn.Cell):
    def __init__(self, msg):
        super(PrintGradShapeTypeCell, self).__init__()
        self.print_shape_type = P.InsertGradientOf(PrintShapeType(msg))

    def construct(self, x):
        self.print_shape_type(x)
        return x
