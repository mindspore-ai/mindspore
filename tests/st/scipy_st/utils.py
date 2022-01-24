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
"""utility functions for mindspore.scipy st tests"""
from typing import List

import numpy as onp
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore.common import dtype as mstype


def to_tensor(obj, dtype=None):
    if dtype is None:
        res = Tensor(obj)
        if res.dtype == mnp.float64:
            res = res.astype(mnp.float32)
        if res.dtype == mnp.int64:
            res = res.astype(mnp.int32)
    else:
        res = Tensor(obj, dtype)
    return res


def match_array(actual, expected, error=0, err_msg=''):
    if isinstance(actual, int):
        actual = onp.asarray(actual)

    if isinstance(expected, (int, tuple)):
        expected = onp.asarray(expected)

    if error > 0:
        onp.testing.assert_almost_equal(actual, expected, decimal=error, err_msg=err_msg)
    else:
        onp.testing.assert_equal(actual, expected, err_msg=err_msg)


def match_matrix(actual, expected, error=0, err_msg=''):
    if actual.shape != expected.shape:
        raise ValueError(
            err_msg.join(f" actual shape {actual.shape} is not equal to expected input shape {expected.shape}"))
    sub_abs = mnp.abs(mnp.subtract(actual, expected))
    no_zero_max = sub_abs.max()
    if no_zero_max > Tensor(error, dtype=mstype.float64):
        raise ValueError(
            err_msg.join(f" actual value: {actual} is not equal to expected input value: {expected}"))


def create_full_rank_matrix(shape, dtype):
    if len(shape) < 2 or shape[-1] != shape[-2]:
        raise ValueError(
            'Full rank matrix must be a square matrix, but has shape: ', shape)

    invertible = False
    a = None
    while not invertible:
        a = onp.random.random(shape).astype(dtype)
        try:
            onp.linalg.inv(a)
            invertible = True
        except onp.linalg.LinAlgError:
            pass

    return a


def create_random_rank_matrix(shape, dtype):
    if len(shape) < 2:
        raise ValueError(
            'random rank matrix must shape bigger than two dims, but has shape: ', shape)
    return onp.random.random(shape).astype(dtype)


def create_sym_pos_matrix(shape, dtype):
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(
            'Symmetric positive definite matrix must be a square matrix, but has shape: ', shape)

    n = shape[-1]
    a = (onp.random.random(shape) + onp.eye(n)).astype(dtype)
    return onp.dot(a, a.T)


def gradient_check(x, net, epsilon=1e-3):
    # some utils
    def _tensor_to_numpy(arg: List[Tensor]) -> List[onp.ndarray]:
        return [_arg.asnumpy() for _arg in arg]

    def _numpy_to_tensor(arg: List[onp.ndarray]) -> List[Tensor]:
        return [Tensor(_arg) for _arg in arg]

    def _add_value(arg: List[onp.ndarray], outer, inner, value):
        arg[outer][inner] += value
        return arg

    def _flatten(arg: List[onp.ndarray]) -> onp.ndarray:
        arg = [_arg.reshape((-1,)) for _arg in arg]
        return onp.concatenate(arg)

    if isinstance(x, Tensor):
        x = [x]

    # using automatic differentiation to calculate gradient
    grad_net = ops.GradOperation(get_all=True)(net)
    x_grad = grad_net(*x)
    x_grad = _tensor_to_numpy(x_grad)

    # using the definition of a derivative to calculate gradient
    x = _tensor_to_numpy(x)
    x_grad_approx = [onp.zeros_like(_x) for _x in x_grad]
    for outer, _x in enumerate(x):
        for inner, _ in onp.ndenumerate(_x):
            x = _add_value(x, outer, inner, epsilon)
            y_plus = net(*_numpy_to_tensor(x)).asnumpy()

            x = _add_value(x, outer, inner, -2 * epsilon)
            y_minus = net(*_numpy_to_tensor(x)).asnumpy()

            y_grad = (y_plus - y_minus) / (2 * epsilon)
            x = _add_value(x, outer, inner, epsilon)
            x_grad_approx = _add_value(x_grad_approx, outer, inner, y_grad)

    x_grad = _flatten(x_grad)
    x_grad_approx = _flatten(x_grad_approx)
    numerator = onp.linalg.norm(x_grad - x_grad_approx)
    denominator = onp.linalg.norm(x_grad) + onp.linalg.norm(x_grad_approx)
    difference = numerator / denominator
    return difference
