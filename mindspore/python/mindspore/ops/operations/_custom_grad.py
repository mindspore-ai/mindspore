# Copyright 2021-2022 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Register bprop function for Custom Hybrid Autodiff"""

from __future__ import absolute_import
from mindspore import log as logger


class Registry:
    """
    Registry class for custom bprop function.
    Key: length of signatures
    Value : bprop function for custom hybrid op
    """

    def __init__(self):
        self._bprops = {}

    @classmethod
    def instance(cls):
        """
        Get singleton of Registry.

        Returns:
            An instance of Registry.
        """
        if not hasattr(Registry, "_instance"):
            Registry._instance = Registry()
        return Registry._instance

    @staticmethod
    def register(sig_number):
        """register length of signatures, bprop function to dict"""
        def deco(fn):
            """deco function"""
            Registry.instance().set(sig_number, fn)
            return fn
        return deco

    def get(self, sig_number):
        """get bprop function by length of signatures"""
        if sig_number not in self._bprops:
            logger.error(f"Autodiff currently doesn't support hyrbrid function with input num :{sig_number}. \
                Supported input num is from 1 to 10")
        fn = self._bprops.get(sig_number)
        return fn

    def set(self, sig_number, fn):
        """set length of signatures, bprop function to dict"""
        self._bprops[sig_number] = fn


def autodiff_bprop(n):
    """get bprop function"""
    return Registry.instance().get(n)


def get_outs(out, dout):
    """combine out and dout to a tuple"""
    if isinstance(out, tuple):
        tupleout = out
    else:
        tupleout = (out,)
    if isinstance(dout, tuple):
        tupledout = dout
    else:
        tupledout = (dout,)
    return tupleout + tupledout


@Registry.register(1)
def bprop_one(op):
    """bprop func for custom op with one input"""
    def bprop(x1, out, dout):
        inputs = (x1,) + get_outs(out, dout)
        res = op(*inputs)
        return res
    return bprop


@Registry.register(2)
def bprop_two(op):
    """bprop func for custom op with two inputs"""
    def bprop(x1, x2, out, dout):
        inputs = (x1, x2) + get_outs(out, dout)
        res = op(*inputs)
        return res
    return bprop


@Registry.register(3)
def bprop_three(op):
    """bprop func for custom op with three inputs"""
    def bprop(x1, x2, x3, out, dout):
        inputs = (x1, x2, x3) + get_outs(out, dout)
        res = op(*inputs)
        return res
    return bprop


@Registry.register(4)
def bprop_four(op):
    """bprop func for custom op with four inputs"""
    def bprop(x1, x2, x3, x4, out, dout):
        inputs = (x1, x2, x3, x4) + get_outs(out, dout)
        res = op(*inputs)
        return res
    return bprop


@Registry.register(5)
def bprop_five(op):
    """bprop func for custom op with five inputs"""
    def bprop(x1, x2, x3, x4, x5, out, dout):
        inputs = (x1, x2, x3, x4, x5) + get_outs(out, dout)
        res = op(*inputs)
        return res
    return bprop


@Registry.register(6)
def bprop_six(op):
    """bprop func for custom op with six inputs"""
    def bprop(x1, x2, x3, x4, x5, x6, out, dout):
        inputs = (x1, x2, x3, x4, x5, x6) + get_outs(out, dout)
        res = op(*inputs)
        return res
    return bprop


@Registry.register(7)
def bprop_seven(op):
    """bprop func for custom op with seven inputs"""
    def bprop(x1, x2, x3, x4, x5, x6, x7, out, dout):
        inputs = (x1, x2, x3, x4, x5, x6, x7) + get_outs(out, dout)
        res = op(*inputs)
        return res
    return bprop


@Registry.register(8)
def bprop_eight(op):
    """bprop func for custom op with eight inputs"""
    def bprop(x1, x2, x3, x4, x5, x6, x7, x8, out, dout):
        inputs = (x1, x2, x3, x4, x5, x6, x7, x8) + get_outs(out, dout)
        res = op(*inputs)
        return res
    return bprop


@Registry.register(9)
def bprop_nine(op):
    """bprop func for custom op with nine inputs"""
    def bprop(x1, x2, x3, x4, x5, x6, x7, x8, x9, out, dout):
        inputs = (x1, x2, x3, x4, x5, x6, x7, x8, x9) + get_outs(out, dout)
        res = op(*inputs)
        return res
    return bprop


@Registry.register(10)
def bprop_ten(op):
    """bprop func for custom op with ten inputs"""
    def bprop(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, out, dout):
        inputs = (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) + get_outs(out, dout)
        res = op(*inputs)
        return res
    return bprop
