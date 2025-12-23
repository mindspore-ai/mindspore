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

from types import MethodType
import weakref
import uuid
from collections import OrderedDict
from collections import defaultdict
from mindspore import log as logger
from mindspore.nn.cell import Cell
from mindspore.common.tensor import Tensor
from mindspore import ops
from mindspore.ops.composite import GradOperation
from mindspore.common._register_for_recompute import recompute_registry
from mindspore.common.api import _pynative_executor, _no_grad, _run_in_jit, saved_tensors_hooks
from mindspore.common.generator import get_rng_state, set_rng_state
from mindspore.common._grad_function import _Function
from mindspore.train.amp import AmpDecorator
from mindspore._c_expression.amp import get_curr_amp_strategy


class _WrapCell(Cell):
    """
    The warp cell is used by recompute cell,
    which can set mixed precision to warp cell
    """

    def __init__(self, function):
        super().__init__(auto_prefix=False)
        self.function = function

    def construct(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class _InputSaver(_Function):
    """
    A custom function saver for recompute inputs, only support tensor.
    """
    @staticmethod
    def forward(ctx, *args):
        tensor_idx, tensors = zip(*[(idx, t) for idx, t in enumerate(args) if isinstance(t, Tensor)])
        idx2tensoridx = {idx: saved_tensor_idx for saved_tensor_idx, idx in enumerate(tensor_idx)}
        new_args = [None if isinstance(arg, Tensor) else arg for arg in args]

        def recover_inputs(saved_inputs):
            res = []
            for index, t in enumerate(new_args):
                if index in tensor_idx:
                    res.append(saved_inputs[idx2tensoridx[index]])
                else:
                    res.append(t)
            return res[1:]
        ctx.recover_inputs = recover_inputs
        ctx.save_for_backward(*tensors)
        return ops.stop_gradient(args[0])

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise RuntimeError("_InputSaver backward function should not be called.")


class _OutputSaver(_Function):
    """
    A custom function saver for recompute outputs, Only support tensor.
    """
    @staticmethod
    def forward(ctx, *args):
        saved_tensors = []
        new_args = []
        for t in args:
            if isinstance(t, Tensor):
                new_tensor = ops.stop_gradient(t)
                saved_tensors.append(new_tensor)
                new_args.append(new_tensor)
            else:
                raise TypeError("should be tensor")
        ctx.save_for_backward(*saved_tensors)
        return new_args[0]

    @staticmethod
    def backward(ctx, *grad_outputs):
        return grad_outputs


class _RecomputeState:
    """
    Record recompute temp state.
    """
    def __init__(self, recompute_fn):
        self.input_node = None
        self.hybrid_args = None
        self.kwargs = None
        self.recompute_fn = recompute_fn
        self.placeholders = []
        self.placeholder_counter = defaultdict(int)
        self.recomputed_data = defaultdict(weakref.WeakKeyDictionary)
        self.is_recomputed = defaultdict(bool)

    def wrap_original_inputs(self, *args, **kwargs):
        """Wrap and store original inputs for later recovery during recomputation."""
        self.kwargs = kwargs
        # pylint: disable=protected-access
        self.hybrid_args = [t._grad_node if isinstance(t, Tensor) and t._grad_node is not None
                            and isinstance(t._grad_node, _OutputSaver) else t for t in args]

    def unwrap_original_inputs(self):
        """Unwrap the stored inputs by recovering saved tensors from _OutputSaver nodes."""
        new_args = [self.kwargs]
        for t in self.hybrid_args:
            if isinstance(t, _OutputSaver):
                new_args.append(t.saved_tensors[0])
            else:
                new_args.append(t)
        return new_args

    def recover_inputs(self):
        """Recover inputs from saved state for recomputation during backward pass."""
        if self.input_node is not None:
            # pylint: disable=protected-access
            ctx = self.input_node._grad_node
            return ctx.recover_inputs(ctx.saved_tensors)
        return self.unwrap_original_inputs()


class _PlaceHolder:
    """
    Placeholder
    """
    def __init__(self):
        pass


class _CreatePlaceholderHook(saved_tensors_hooks):
    """
    Placeholder hook for forward function which need recompute.
    """
    def __init__(self, state: _RecomputeState):
        def pack(x):
            holder = _PlaceHolder()
            state.placeholders.append(weakref.ref(holder))
            return holder

        def unpack(holder):
            engine_id = _pynative_executor.get_current_autodiff_engine_id()
            if engine_id == -1:
                engine_id = int(uuid.uuid4())
            if not state.is_recomputed[engine_id]:
                new_inputs = state.recover_inputs()
                prev_grad_flag = _pynative_executor.grad_flag()
                _pynative_executor.set_grad_flag(True)
                with _RecomputationHook(weakref.ref(state), engine_id):
                    state.recompute_fn(*new_inputs)
                _pynative_executor.set_grad_flag(prev_grad_flag)
                state.is_recomputed[engine_id] = True
            if state.recomputed_data[engine_id].get(holder, default=None) is None:
                raise RuntimeError("Unpack is being triggered for a tensor, make sure to do this only once!")
            val = state.recomputed_data[engine_id][holder]
            state.recomputed_data[engine_id].pop(holder)
            return val
        super().__init__(pack, unpack)


class _RecomputationHook(saved_tensors_hooks):
    """
    Recompute hook for saved tensor which need get activation value.
    """
    def __init__(self, weak_state, engine_id):
        def pack(x):
            # pylint: disable=protected-access
            x = ops.stop_gradient(x) if x._requires_grad else x
            state = weak_state()
            if state is None:
                raise RuntimeError("Recompute state has been garbage collected unexpectedly.")
            current_idx = state.placeholder_counter[engine_id]
            state.placeholder_counter[engine_id] += 1
            if current_idx >= len(state.placeholders):
                raise RuntimeError('This forward function contains non-determinism and is non-reentrant, '
                                   'which cannot be recomputed.')
            placeholder = state.placeholders[current_idx]()
            if placeholder is not None:
                state.recomputed_data[engine_id][placeholder] = x
            return x

        def unpack(x):
            return x
        super().__init__(pack, unpack)


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
        super().__init__(auto_prefix=False)
        self.args = []
        self.kwargs = []
        self.wrap_cell = _WrapCell(block)
        self.wrap_cell.set_inputs()

        self.net = block
        self.internal_params = []
        self.save_rng_state = False
        self.cpu_rng_state = None
        self._add_attr("is_cell_recompute", "True")
        self.grad = GradOperation(get_all=True, get_by_list=True, sens_param=True)
        self.init_mixed_precision_type(block)
        self.amp_strategy = None

    def construct(self, *args, **kwargs):
        _check_input_args_validate(self.net, args, kwargs)
        self.args.append(args)
        self.kwargs.append(kwargs)
        self.save_rng_state = kwargs.pop("save_rng_state", True)
        if self.save_rng_state:
            self.cpu_rng_state = get_rng_state()
        self.amp_strategy = get_curr_amp_strategy()
        with _no_grad():
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
            input_args_for_check = list(input_args) + list(kwargs.values())
        else:
            input_args_for_check = list(input_args)
        # To detach inputs to avoid erasing auto grad meta info of origin inputs.
        input_args = _detach_input(input_args)
        kwargs = _detach_input(kwargs)
        kwargs['sens'] = grad_input
        try:
            pre_rng_state = get_rng_state()
            set_rng_state(self.cpu_rng_state)
            _pynative_executor.set_is_run_recompute(True)
            if self.amp_strategy:
                with AmpDecorator(self.amp_strategy.get_amp_level(), self.amp_strategy.get_amp_dtype(),
                                  self.amp_strategy.get_white_list(), self.amp_strategy.get_black_list()):
                    grads = self.grad(self.net, self.internal_params)(*input_args, **kwargs)
            else:
                grads = self.grad(self.net, self.internal_params)(*input_args, **kwargs)
            _pynative_executor.set_is_run_recompute(False)
            set_rng_state(pre_rng_state)
        except Exception as err:
            _pynative_executor.clear_res()
            raise err
        weights = OrderedDict()
        input_grads = list(grads[0])
        _padding_input_grads(input_args_for_check, input_grads)
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


def _check_input_args_validate(block, args, kwargs):
    """
    Check recompute input args validate
    :param args:
    :return:
    """
    if not (any(isinstance(arg, Tensor) for arg in args) or
            any(isinstance(arg, Tensor) for arg in kwargs.values())):
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
            if all(not isinstance(data, Tensor) for data in arg):
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
    """
    Detach input
    :param input_arg:
    :return: detach output
    """
    if isinstance(input_arg, Tensor):
        return ops.stop_gradient(input_arg)
    if isinstance(input_arg, (list, tuple)):
        detach_inputs = []
        for arg in input_arg:
            detach_inputs.append(_detach_input(arg))
        return detach_inputs if isinstance(input_arg, list) else tuple(detach_inputs)
    if isinstance(input_arg, dict):
        detach_inputs = {}
        for key, val in input_arg.items():
            if isinstance(val, Tensor):
                detach_inputs[key] = ops.stop_gradient(val)
            else:
                detach_inputs[key] = val
        return detach_inputs
    return input_arg


def _check_validation(block, use_reentrant):
    if not isinstance(block, Cell) and use_reentrant:
        raise TypeError("Recompute function now only support block which inherited from Cell when use_reentrant=True!")


def recompute(block, *args, use_reentrant=True, output_recompute=False, **kwargs):
    r"""
    This function is used to reduce memory, when run block, rather than
    storing the intermediate activation computed in forward pass, we will recompute it in backward pass.

    Note:
        Recompute function only support block which inherited from Cell object.

    Args:
        block (Cell): Block to be recompute.
        args(tuple): Inputs for block object to run forward pass.

    Keyword Arguments:
        use_reentrant (bool, optional): This keyword is only valid in PyNative mode.
            If use_reentrant=True is set, we will implement recomputation through
            a custom bprop function, which does not support differentiation of complex types
            such as List/Tuple, If use_reentrant=False is set, we will use the saved_tensors_hook functionality
            to implement recomputation, which supports differentiation of tensors inside complex types.
            Default: ``True``.
        output_recompute (bool, optional): This keyword is only valid in PyNative mode. If output_recompute=True is
            set, we will implement recomputation by saved_tensors_hook functionality by default. The output of this cell
            or function will not be stored by subsequent operations for backward. when there are two adjacent cells both
            requiring recomputation (where the output of one cell serves as the input to the other), the recomputation
            of these two cells will be merged. In this case, the output activation values of the first cell will not be
            stored. If output_recompute=False, we will not merge adjacent cells. Default: ``False``.
        \*\*kwargs: Other arguments.

    Returns:
        Same as return type of block.

    Raises:
        TypeError: If `block` is not Cell object.

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
    if _run_in_jit():  # @jit.cond: True
        return ops.recompute_block(block)(*args, **kwargs)
    if output_recompute:
        use_reentrant = False
    _check_validation(block, use_reentrant)
    if use_reentrant:
        return _RecomputeCell(block)(*args, **kwargs)
    return recompute_without_reentrant(block, output_recompute, *args, **kwargs)


def recompute_without_reentrant(block, output_recompute, *args, **kwargs):
    """
    Compute block by recompute function using saved tensors hook.
    :param block:
    :param output_recompute:
    :param args:
    :param kwargs:
    :return:
    """
    save_rng_state = kwargs.pop("save_rng_state", True)
    pre_rng_state = get_rng_state()
    amp_strategy = get_curr_amp_strategy()

    def wrapper_block(*args, **kwargs):
        out = block(*args, **kwargs)
        if not output_recompute:
            return out
        if isinstance(out, Tensor):
            res = _OutputSaver.apply(out)
            return res
        if isinstance(out, list):
            return [_OutputSaver.apply(t) if isinstance(t, Tensor) else t for t in out]
        if isinstance(out, tuple):
            res = [_OutputSaver.apply(t) if isinstance(t, Tensor) else t for t in out]
            return tuple(res)
        return out

    def recompute_function(*inputs):
        kwargs, *args = inputs
        cur_rng_state = get_rng_state()
        if save_rng_state:
            set_rng_state(pre_rng_state)
        prev_grad_flag = _pynative_executor.grad_flag()
        _pynative_executor.set_grad_flag(True)
        if amp_strategy:
            with AmpDecorator(amp_strategy.get_amp_level(), amp_strategy.get_amp_dtype(),
                              amp_strategy.get_white_list(), amp_strategy.get_black_list()):
                wrapper_block(*args, **kwargs)
        else:
            wrapper_block(*args, **kwargs)
        set_rng_state(cur_rng_state)
        _pynative_executor.set_grad_flag(prev_grad_flag)

    if not _pynative_executor.enable_grad():
        return block(*args, **kwargs)

    state = _RecomputeState(recompute_function)
    if _pynative_executor.is_saved_tensor_hook_active():
        fake_val = ops.zeros((0,))
        fake_val.requires_grad_()
        state.input_node = _InputSaver.apply(fake_val, kwargs, *args)
    else:
        state.wrap_original_inputs(*args, **kwargs)

    with _CreatePlaceholderHook(state):
        return wrapper_block(*args, **kwargs)


def recompute_generator(block, use_reentrant=True, output_recompute=False):
    """
    generator of recompute object.
    :param output_recompute:
    :param use_reentrant:
    :param block:
    :return:
    """
    if use_reentrant:
        return _RecomputeCell(block)

    def create_recompute_func(*args, **kwargs):
        return recompute_without_reentrant(block, output_recompute, *args, **kwargs)
    return create_recompute_func


recompute_registry.register(recompute_generator)

__all__ = ['recompute']
__all__.sort()
