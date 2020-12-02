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
"""stencil operations kernel"""

import mindspore.nn as nn
from mindspore.ops import operations as P


class axb_kernel(nn.Cell):
    """create axb_kernel"""
    def __init__(self):
        super(axb_kernel, self).__init__()
        self.pad = P.Pad(((1, 0), (0, 0), (0, 0)))
        self.slice = P.Slice()
        self.shape = P.Shape()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.slice(x1, (0, 0, 0), x_shape)
        out = 0.5 * (x + x1)
        return out


class ayb_kernel(nn.Cell):
    """create ayb_kernel"""
    def __init__(self):
        super(ayb_kernel, self).__init__()
        self.pad = P.Pad(((0, 0), (1, 0), (0, 0)))
        self.slice = P.Slice()
        self.shape = P.Shape()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.slice(x1, (0, 0, 0), x_shape)
        out = 0.5 * (x + x1)
        return out


class azb_kernel(nn.Cell):
    """create azb_kernel"""
    def __init__(self):
        super(azb_kernel, self).__init__()
        self.pad = P.Pad(((0, 0), (0, 0), (1, 0)))
        self.slice = P.Slice()
        self.shape = P.Shape()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.slice(x1, (0, 0, 0), x_shape)
        out = 0.5 * (x + x1)
        return out


class axf_kernel(nn.Cell):
    """create axf_kernel"""
    def __init__(self):
        super(axf_kernel, self).__init__()
        self.pad = P.Pad(((0, 1), (0, 0), (0, 0)))
        self.slice = P.Slice()
        self.shape = P.Shape()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.slice(x1, (1, 0, 0), x_shape)
        out = 0.5 * (x + x1)
        return out


class ayf_kernel(nn.Cell):
    """create ayf_kernel"""
    def __init__(self):
        super(ayf_kernel, self).__init__()
        self.pad = P.Pad(((0, 0), (0, 1), (0, 0)))
        self.slice = P.Slice()
        self.shape = P.Shape()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.slice(x1, (0, 1, 0), x_shape)
        out = 0.5 * (x + x1)
        return out


class azf_kernel(nn.Cell):
    """create azf_kernel"""
    def __init__(self):
        super(azf_kernel, self).__init__()
        self.pad = P.Pad(((0, 0), (0, 0), (0, 1)))
        self.slice = P.Slice()
        self.shape = P.Shape()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.slice(x1, (0, 0, 1), x_shape)
        out = 0.5 * (x + x1)
        return out


class dxb_kernel(nn.Cell):
    """create dxb_kernel"""
    def __init__(self):
        super(dxb_kernel, self).__init__()
        self.pad = P.Pad(((1, 0), (0, 0), (0, 0)))
        self.slice = P.Slice()
        self.shape = P.Shape()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.slice(x1, (0, 0, 0), x_shape)
        x = x - x1
        return x


class dxf_kernel(nn.Cell):
    """create dxf_kernel"""
    def __init__(self):
        super(dxf_kernel, self).__init__()
        self.pad = P.Pad(((0, 1), (0, 0), (0, 0)))
        self.slice = P.Slice()
        self.shape = P.Shape()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.slice(x1, (1, 0, 0), x_shape)
        x = x1 - x
        return x


class dyb_kernel(nn.Cell):
    """create dyb_kernel"""
    def __init__(self):
        super(dyb_kernel, self).__init__()
        self.pad = P.Pad(((0, 0), (1, 0), (0, 0)))
        self.slice = P.Slice()
        self.shape = P.Shape()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.slice(x1, (0, 0, 0), x_shape)
        x = x - x1
        return x


class dyf_kernel(nn.Cell):
    """create dyf_kernel"""
    def __init__(self):
        super(dyf_kernel, self).__init__()
        self.pad = P.Pad(((0, 0), (0, 1), (0, 0)))
        self.slice = P.Slice()
        self.shape = P.Shape()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.slice(x1, (0, 1, 0), x_shape)
        x = x1 - x
        return x


class dzb_kernel(nn.Cell):
    """create dzb_kernel"""
    def __init__(self):
        super(dzb_kernel, self).__init__()
        self.pad = P.Pad(((0, 0), (0, 0), (1, 0)))
        self.slice = P.Slice()
        self.shape = P.Shape()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.slice(x1, (0, 0, 0), x_shape)
        x = x - x1
        return x


class dzf_kernel(nn.Cell):
    """create dzf_kernel"""
    def __init__(self):
        super(dzf_kernel, self).__init__()
        self.pad = P.Pad(((0, 0), (0, 0), (0, 1)))
        self.slice = P.Slice()
        self.shape = P.Shape()

    def construct(self, x):
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.slice(x1, (0, 0, 1), x_shape)
        x = x1 - x
        return x
