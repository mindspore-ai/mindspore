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
"""Utitly functions to help distribution class."""
import numpy as np
from mindspore import context
from mindspore._checkparam import Validator as validator
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype
from mindspore.ops import _utils as utils
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops.primitive import constexpr, PrimitiveWithInfer, prim_attr_register
import mindspore.nn as nn
import mindspore.nn.probability as msp


def cast_to_tensor(t, hint_type=mstype.float32):
    """
    Cast an user input value into a Tensor of dtype.
    If the input t is of type Parameter, t is directly returned as a Parameter.

    Args:
        t (int, float, list, numpy.ndarray, Tensor, Parameter): object to be cast to Tensor.
        dtype (mindspore.dtype): dtype of the Tensor. Default: mstype.float32.

    Raises:
        RuntimeError: if t cannot be cast to Tensor.

    Returns:
        Tensor.
    """
    if t is None:
        raise ValueError(f'Input cannot be None in cast_to_tensor')
    if isinstance(t, Parameter):
        return t
    t_type = hint_type
    if isinstance(t, Tensor):
        # convert the type of tensor to dtype
        return Tensor(t.asnumpy(), dtype=t_type)
    if isinstance(t, (list, np.ndarray)):
        return Tensor(t, dtype=t_type)
    if isinstance(t, bool):
        raise TypeError(f'Input cannot be Type Bool')
    if isinstance(t, (int, float)):
        return Tensor(t, dtype=t_type)
    invalid_type = type(t)
    raise TypeError(
        f"Unable to convert input of type {invalid_type} to a Tensor of type {t_type}")


def convert_to_batch(t, batch_shape, required_type):
    """
    Convert a Tensor to a given batch shape.

    Args:
        t (int, float, list, numpy.ndarray, Tensor, Parameter): Tensor to be converted.
        batch_shape (tuple): desired batch shape.
        dtype (mindspore.dtype): desired dtype.

    Raises:
        RuntimeError: if the converison cannot be done.

    Returns:
        Tensor, with shape of batch_shape.
    """
    if isinstance(t, Parameter):
        return t
    t = cast_to_tensor(t, required_type)
    return Tensor(np.broadcast_to(t.asnumpy(), batch_shape), dtype=required_type)


def cast_type_for_device(dtype):
    """
    use the alternative dtype supported by the device.
    Args:
        dtype (mindspore.dtype): input dtype.
    Returns:
        mindspore.dtype.
    """
    if context.get_context("device_target") == "GPU":
        if dtype in mstype.uint_type or dtype == mstype.int8:
            return mstype.int16
        if dtype == mstype.int64:
            return mstype.int32
        if dtype == mstype.float64:
            return mstype.float32
    return dtype


def check_scalar_from_param(params):
    """
    Check if params are all scalars.

    Args:
        params (dict): parameters used to initialize distribution.

    Notes: String parameters are excluded.
    """
    for value in params.values():
        if value is None:
            continue
        if isinstance(value, (msp.bijector.Bijector, msp.distribution.Distribution)):
            return params['distribution'].is_scalar_batch
        if isinstance(value, Parameter):
            return False
        if not isinstance(value, (int, float, str, type(params['dtype']))):
            return False
    return True


def calc_broadcast_shape_from_param(params):
    """
    Calculate the broadcast shape from params.

    Args:
        params (dict): parameters used to initialize distribution.

    Returns:
        tuple.
    """
    broadcast_shape = []
    for value in params.values():
        if isinstance(value, (msp.bijector.Bijector, msp.distribution.Distribution)):
            return params['distribution'].broadcast_shape
        if isinstance(value, (str, type(params['dtype']))):
            continue
        if value is None:
            return None
        if isinstance(value, Parameter):
            value_t = value.data
        else:
            value_t = cast_to_tensor(value, mstype.float32)
        broadcast_shape = utils.get_broadcast_shape(
            broadcast_shape, list(value_t.shape), params['name'])
    return tuple(broadcast_shape)


def check_greater_equal_zero(value, name):
    """
    Check if the given Tensor is greater zero.

    Args:
        value (Tensor, Parameter): value to be checked.
        name (str) : name of the value.

    Raises:
        ValueError: if the input value is less than zero.

    """
    if isinstance(value, Parameter):
        if not isinstance(value.data, Tensor):
            return
        value = value.data
    comp = np.less(value.asnumpy(), np.zeros(value.shape))
    if comp.any():
        raise ValueError(f'{name} should be greater than ot equal to zero.')


def check_greater_zero(value, name):
    """
    Check if the given Tensor is strictly greater than zero.

    Args:
        value (Tensor, Parameter): value to be checked.
        name (str) : name of the value.

    Raises:
        ValueError: if the input value is less than or equal to zero.

    """
    if value is None:
        raise ValueError(f'input value cannot be None in check_greater_zero')
    if isinstance(value, Parameter):
        if not isinstance(value.data, Tensor):
            return
        value = value.data
    comp = np.less(np.zeros(value.shape), value.asnumpy())
    if not comp.all():
        raise ValueError(f'{name} should be greater than zero.')


def check_greater(a, b, name_a, name_b):
    """
    Check if Tensor b is strictly greater than Tensor a.

    Args:
        a (Tensor, Parameter): input tensor a.
        b (Tensor, Parameter): input tensor b.
        name_a (str): name of Tensor_a.
        name_b (str): name of Tensor_b.

    Raises:
        ValueError: if b is less than or equal to a
    """
    if a is None or b is None:
        raise ValueError(f'input value cannot be None in check_greater')
    if isinstance(a, Parameter) or isinstance(b, Parameter):
        return
    comp = np.less(a.asnumpy(), b.asnumpy())
    if not comp.all():
        raise ValueError(f'{name_a} should be less than {name_b}')


def check_prob(p):
    """
    Check if p is a proper probability, i.e. 0 < p <1.

    Args:
        p (Tensor, Parameter): value to be checked.

    Raises:
        ValueError: if p is not a proper probability.
    """
    if p is None:
        raise ValueError(f'input value cannot be None in check_greater_zero')
    if isinstance(p, Parameter):
        if not isinstance(p.data, Tensor):
            return
        p = p.data
    comp = np.less(np.zeros(p.shape), p.asnumpy())
    if not comp.all():
        raise ValueError('Probabilities should be greater than zero')
    comp = np.greater(np.ones(p.shape), p.asnumpy())
    if not comp.all():
        raise ValueError('Probabilities should be less than one')


def logits_to_probs(logits, is_binary=False):
    """
    converts logits into probabilities.
    Args:
        logits (Tensor)
        is_binary (bool)
    """
    if is_binary:
        return nn.Sigmoid()(logits)
    return nn.Softmax(axis=-1)(logits)


def clamp_probs(probs):
    """
    clamp probs boundary
    Args:
        probs (Tensor)
    """
    eps = P.Eps()(probs)
    return C.clip_by_value(probs, eps, 1-eps)


def probs_to_logits(probs, is_binary=False):
    """
    converts probabilities into logits.
        Args:
        probs (Tensor)
        is_binary (bool)
    """
    ps_clamped = clamp_probs(probs)
    if is_binary:
        return P.Log()(ps_clamped) - P.Log()(1-ps_clamped)
    return P.Log()(ps_clamped)


def check_type(data_type, value_type, name):
    if not data_type in value_type:
        raise TypeError(
            f"For {name}, valid type include {value_type}, {data_type} is invalid")


@constexpr
def raise_none_error(name):
    raise TypeError(f"the type {name} should be subclass of Tensor."
                    f" It should not be None since it is not specified during initialization.")

@constexpr
def raise_probs_logits_error():
    raise TypeError("Either 'probs' or 'logits' must be specified, but not both.")

@constexpr
def raise_broadcast_error(shape_a, shape_b):
    raise ValueError(f"Shape {shape_a} and {shape_b} is not broadcastable.")

@constexpr
def raise_not_impl_error(name):
    raise ValueError(
        f"{name} function should be implemented for non-linear transformation")


@constexpr
def check_distribution_name(name, expected_name):
    if name is None:
        raise ValueError(
            f"Input dist should be a constant which is not None.")
    if name != expected_name:
        raise ValueError(
            f"Expected dist input is {expected_name}, but got {name}.")


class CheckTuple(PrimitiveWithInfer):
    """
    Check if input is a tuple.
    """
    @prim_attr_register
    def __init__(self):
        super(CheckTuple, self).__init__("CheckTuple")
        self.init_prim_io_names(inputs=['x', 'name'], outputs=['dummy_output'])

    def __infer__(self, x, name):
        if not isinstance(x['dtype'], tuple):
            raise TypeError(
                f"For {name['value']}, Input type should b a tuple.")

        out = {'shape': None,
               'dtype': None,
               'value': x["value"]}
        return out

    def __call__(self, x, name):
        if context.get_context("mode") == 0:
            return x["value"]
        # Pynative mode
        if isinstance(x, tuple):
            return x
        raise TypeError(f"For {name}, input type should be a tuple.")


class CheckTensor(PrimitiveWithInfer):
    """
    Check if input is a Tensor.
    """
    @prim_attr_register
    def __init__(self):
        super(CheckTensor, self).__init__("CheckTensor")
        self.init_prim_io_names(inputs=['x', 'name'], outputs=['dummy_output'])

    def __infer__(self, x, name):
        src_type = x['dtype']
        validator.check_subclass(
            "input", src_type, [mstype.tensor], name["value"])

        out = {'shape': None,
               'dtype': None,
               'value': None}
        return out

    def __call__(self, x, name):
        if isinstance(x, Tensor):
            return x
        raise TypeError(f"For {name}, input type should be a Tensor or Parameter.")

def set_param_type(args, hint_type):
    """
    Find the common type among arguments.

    Args:
        args (dict): dictionary of arguments, {'name':value}.
        hint_type (mindspore.dtype): hint type to return.

    Raises:
        TypeError: if tensors in args are not the same dtype.
    """
    common_dtype = None
    for name, arg in args.items():
        if hasattr(arg, 'dtype'):
            if isinstance(arg, np.ndarray):
                cur_dtype = mstype.pytype_to_dtype(arg.dtype)
            else:
                cur_dtype = arg.dtype
            if common_dtype is None:
                common_dtype = cur_dtype
            elif cur_dtype != common_dtype:
                raise TypeError(f"{name} should have the same dtype as other arguments.")
    int_type = mstype.int_type + mstype.uint_type
    if common_dtype in int_type or common_dtype == mstype.float64:
        return mstype.float32
    return hint_type if common_dtype is None else common_dtype
