# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Defines other operators with functional form."""

from collections import OrderedDict
from types import MethodType
from mindspore import log as logger
from mindspore.nn.cell import Cell
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore import ops
from mindspore.ops.composite import GradOperation
from mindspore.common._register_for_recompute import recompute_registry
from mindspore.common.api import _pynative_executor
from mindspore.common.generator import get_rng_state, set_rng_state


class _WrapCell(Cell):
    """
    The warp cell is used by recompute cell,
    which can set mixed precision to warp cell
    """

    def __init__(self, function):
        super(_WrapCell, self).__init__()
        self.function = function

    def construct(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class _RecomputeCell(Cell):
    """
    Recompute cell, given the sub block, this cell will recompute the block, rather than
    storing the intermediate activation computed in forward pass, we will recompute it in backward pass.
    Note:
     - RecomputeCell now only support pynative mode.
     - When use recompute function, block object should not decorated by @jit.
    """

    def __init__(self, block):
        """Initialize Recompute cell."""
        super(_RecomputeCell, self).__init__(auto_prefix=False)
        self.args = []
        self.kwargs = []
        self.wrap_cell = _WrapCell(block)

        self.net = block
        self.internal_params = []
        self.save_rng_state = False
        self.cpu_rng_state = None
        self._add_attr("is_cell_recompute", "True")
        self.grad = GradOperation(get_all=True, get_by_list=True, sens_param=True)
        self.init_mixed_precision_type(block)

    def construct(self, *args, **kwargs):
        _check_input_args_validate(self.net, args)
        self.args.append(args)
        self.kwargs.append(kwargs)
        self.save_rng_state = kwargs.pop("save_rng_state", True)
        if self.save_rng_state:
            self.cpu_rng_state = get_rng_state()
        return self.net(*args, **kwargs)

    def bprop(self, *args):
        """
        Custom grad method for recompute
        :param args:
        :return: input grad and weight grads
        """
        grad_input = args[-1]
        input_args = self.args[-1]
        kwargs = self.kwargs[-1]
        self.args.pop()
        self.kwargs.pop()
        if kwargs:
            input_args = list(input_args) + list(kwargs.values())
        # To detach inputs to avoid erasing auto grad meta info of origin inputs.
        input_args = _detach_input(input_args)
        try:
            pre_rng_state = get_rng_state()
            set_rng_state(self.cpu_rng_state)
            _pynative_executor.set_is_run_recompute(True)
            grads = self.grad(self.net, self.internal_params)(*input_args, grad_input)
            _pynative_executor.set_is_run_recompute(False)
            set_rng_state(pre_rng_state)
        except Exception as err:
            _pynative_executor.clear_res()
            raise err
        weights = OrderedDict()
        input_grads = list(grads[0])
        _padding_input_grads(input_args, input_grads)
        for i, param in enumerate(self.internal_params):
            weights[param] = grads[1][i]
        return tuple(input_grads), weights

    def init_mixed_precision_type(self, block):
        """
        init mix precision
        :param block:
        :return:
        """
        if isinstance(block, Cell):
            # To avoid sub cell same name
            block.check_names_and_refresh_name()
            self.internal_params = block.trainable_params()
            return
        if isinstance(block, MethodType) and isinstance(block.__self__, Cell):
            # To avoid sub cell same name
            block.__self__.check_names_and_refresh_name()
            self.internal_params = block.__self__.trainable_params()
            self.wrap_cell.mixed_precision_type = block.__self__.get_mixed_precision_type()
            self.wrap_cell.set_mixed_precision_type(block.__self__.get_mixed_precision_type())
            self.net = self.wrap_cell
        else:
            raise TypeError("For Recompute cell, it not support FunctionType function, "
                            "only support Cell object or MethodType function!")


def _check_input_args_validate(block, args):
    """
    Check recompute input args validate
    :param args:
    :return:
    """
    if not any([isinstance(arg, Tensor) for arg in args]):
        logger.warning("None of the inputs of function are tensors, which not need use recompute!")
    for arg in args:
        if isinstance(arg, (tuple, list)):
            for data in arg:
                if isinstance(data, Tensor):
                    logger.info("For recompute block {}, tensor input in Tuple or list "
                                "will not calculate grads!".format(block))
                    break


def _padding_input_grads(args, input_grads):
    """
    Padding input grads to same as input args
    :param args:
    :param input_grads:
    :return:
    """
    for i, arg in enumerate(args):
        if isinstance(arg, (list, tuple)):
            if all([not isinstance(data, Tensor) for data in arg]):
                input_grads.insert(i, None)
            else:
                # None is placeholder
                grads = [None for data in arg]
                input_grads.insert(i, grads)
        elif not isinstance(arg, Tensor):
            input_grads.insert(i, None)
    if len(args) != len(input_grads):
        raise ValueError("For recompute cell, the input grads size should be same as input args size: {}, "
                         "but got {}".format(len(args), len(input_grads)))


def _detach_input(input_arg):
    if isinstance(input_arg, Tensor):
        return ops.stop_gradient(input_arg)
    if isinstance(input_arg, (list, tuple)):
        detach_inputs = []
        for arg in input_arg:
            detach_inputs.append(_detach_input(arg))
        return detach_inputs if isinstance(input_arg, list) else tuple(detach_inputs)
    return input_arg


def _check_validation(block):
    if not isinstance(block, Cell):
        raise TypeError("Recompute function now only support block which inherited from Cell!")
    if context.get_context("mode") != context.PYNATIVE_MODE:
        raise AssertionError("Recompute function now only support pynative mode, you can use "
                             "Cell.recompute() in graph mode.")
    if block.construct.__code__.co_name == "staging_specialize":
        logger.warning('Block\'s construct method decorated by @jit that recompute '
                       'function will not come into effect.')


def recompute(block, *args, **kwargs):
    r"""
    This function is used to reduce memory, when run block, rather than
    storing the intermediate activation computed in forward pass, we will recompute it in backward pass.

    Note:
        - Recompute function only support block which inherited from Cell object.
        - This function interface now only support pynative mode. you can use Cell.recompute interface
          in graph mode.
        - When use recompute function, block object should not decorated by @jit.

    Args:
        block (Cell): Block to be recompute.
        args(tuple): Inputs for block object to run forward pass.
        kwargs(dict): Optional input for recompute function.

    Returns:
        Same as return type of block.

    Raises:
        TypeError: If `block` is not Cell object.
        AssertionError: If execute mode is not PYNATIVE_MODE.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> from mindspore import ops
        >>> from mindspore import Tensor, recompute
        >>> class MyCell(nn.Cell):
        ...     def __init__(self):
        ...         super(MyCell, self).__init__(auto_prefix=False)
        ...         self.conv = nn.Conv2d(2, 2, 2, has_bias=False, weight_init='ones')
        ...         self.relu = ops.ReLU()
        ...
        ...     def construct(self, x):
        ...         y = recompute(self.conv, x)
        ...         return self.relu(y)
        >>> inputs = Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 2)
        >>> my_net = MyCell()
        >>> grad = ops.grad(my_net)(inputs)
        >>> print(grad)
        [[[[2. 4.]
           [4. 8.]]
          [[2. 4.]
           [4. 8.]]]
         [[[2. 4.]
           [4. 8.]]
          [[2. 4.]
           [4. 8.]]]]
    """

    _check_validation(block)
    return _RecomputeCell(block)(*args, **kwargs)


def recompute_generator(block):
    """
    generator of recompute object.
    :param block:
    :return:
    """
    return _RecomputeCell(block)


recompute_registry.register(recompute_generator)

__all__ = ['recompute']
__all__.sort()
