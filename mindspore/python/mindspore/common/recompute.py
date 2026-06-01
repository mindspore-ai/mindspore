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

from types import MethodType  # pylint: disable=wrong-import-position
import weakref
import uuid
import contextlib
import contextvars
from collections import OrderedDict
from collections import defaultdict
from mindspore import log as logger
from mindspore.nn.cell import Cell
from mindspore.common.tensor import Tensor
from mindspore import ops
from mindspore.ops.composite import GradOperation
from mindspore.common._register_for_recompute import recompute_registry
from mindspore.graph.api import _pynative_executor, _no_grad, _run_in_jit, saved_tensors_hooks
from mindspore.common.generator import get_rng_state, set_rng_state
from mindspore.common._grad_function import _Function
from mindspore.train.amp import AmpDecorator
from mindspore._c_expression.amp import get_curr_amp_strategy


def null_context_fn():
    """
    Default context_fn for recompute, returns two null contexts.
    """
    return contextlib.nullcontext(), contextlib.nullcontext()


class _StopRecomputationError(Exception):
    """
    Exception used to early stop recomputation when all needed tensors have been recomputed.
    This is not an error, but a control flow mechanism to avoid unnecessary computation.
    """


# Global flag to control early stopping behavior
_enable_recompute_early_stop = None


class _RecomputeSession:
    """
    Context-local recompute session.

    Callers can force multiple backward engines to use the same session key.
    If retain_on_unpack is True, unpack returns recomputed tensors without
    popping them, allowing a later backward engine to consume them.
    """

    def __init__(self, session_id, retain_on_unpack):
        self.session_id = session_id
        self.retain_on_unpack = retain_on_unpack


_recompute_session = contextvars.ContextVar("_recompute_session", default=None)
_recompute_session_handles = defaultdict(weakref.WeakSet)
_recompute_handle_collector = contextvars.ContextVar("_recompute_handle_collector", default=None)


@contextlib.contextmanager
def _set_recompute_early_stop(enable):
    """
    Context manager that sets whether recompute should stop recomputation early.

    By default, non-reentrant recompute stops recomputation as soon as it
    has computed all needed Tensors. This context manager can be used to disable
    that feature if it is problematic for your specific application.

    This context manager only needs to be active when forward is run. It does
    not need to be active during backward.

    Args:
        enable (bool): Whether to enable early stopping.
    """
    global _enable_recompute_early_stop
    prev = _enable_recompute_early_stop  # pylint: disable=used-before-assignment
    try:
        _enable_recompute_early_stop = enable
        yield
    finally:
        _enable_recompute_early_stop = prev


@contextlib.contextmanager
def _recompute_session_ctx(session_id=None, retain_on_unpack=False):
    """
    Force non-reentrant recompute to use a caller-provided session key.

    Args:
        session_id: Stable key shared by related backward engines. If None, a
            new key is generated and yielded.
        retain_on_unpack: If True, unpack returns recomputed tensors without
            popping them from the session cache.
    """
    if session_id is None:
        session_id = uuid.uuid4().hex
    token = _recompute_session.set(_RecomputeSession(session_id, retain_on_unpack))
    try:
        yield session_id
    finally:
        _recompute_session.reset(token)


def _clear_recompute_session(session_id):
    """Clear retained recompute data for one dx/dw split session."""
    if session_id is None:
        return
    handles = tuple(_recompute_session_handles.pop(session_id, ()))
    for handle in handles:
        handle.clear_session(session_id)


@contextlib.contextmanager
def _recompute_handle_collector_ctx():
    """
    Collect non-reentrant recompute handles created inside this context.

    The yielded list is owned by the caller and remains available after the
    guard exits. The context variable is reset on exit so later recompute calls
    are not collected into this list.
    """
    handles = []
    token = _recompute_handle_collector.set(handles)
    try:
        yield handles
    finally:
        _recompute_handle_collector.reset(token)


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


class _InputSaver(_Function):  # pylint: disable=abstract-method
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


class _OutputSaver(_Function):  # pylint: disable=abstract-method
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


class _RecomputeHandle:
    """
    Record recompute temp handle.
    """
    def __init__(self, recompute_fn, early_stop=True):
        self.input_node = None
        self.hybrid_args = None
        self.kwargs = None
        self.recompute_fn = recompute_fn
        self.placeholders = []
        self.placeholder_counter = defaultdict(int)
        self.recomputed_data = defaultdict(weakref.WeakKeyDictionary)
        self.is_recomputed = defaultdict(bool)
        self.early_stop = early_stop
        self.forward_completed = False

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

    def recompute(self, recompute_key):
        """Run recomputation for the given key when it has not run yet."""
        session = _recompute_session.get()
        if session is not None:
            _recompute_session_handles[recompute_key].add(self)
        if self.is_recomputed[recompute_key]:
            return

        new_inputs = self.recover_inputs()
        prev_grad_flag = _pynative_executor.grad_flag()
        prev_enable_grad = _pynative_executor.enable_grad()
        _pynative_executor.set_grad_flag(True)
        _pynative_executor.set_enable_grad(True)
        try:
            try:
                with _RecomputationHook(weakref.ref(self), recompute_key):
                    _pynative_executor.new_empty_graph()
                    self.recompute_fn(*new_inputs)
                    _pynative_executor.end_empty_graph()
            except _StopRecomputationError:
                _pynative_executor.end_empty_graph()
            self.is_recomputed[recompute_key] = True
        finally:
            _pynative_executor.set_grad_flag(prev_grad_flag)
            _pynative_executor.set_enable_grad(prev_enable_grad)

    def clear_session(self, session_id):
        """Clear retained recompute data associated with a session."""
        self.recomputed_data.pop(session_id, None)
        self.is_recomputed.pop(session_id, None)
        self.placeholder_counter.pop(session_id, None)


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
    def __init__(self, handle: _RecomputeHandle):
        def pack(x):
            holder = _PlaceHolder()
            handle.placeholders.append(weakref.ref(holder))
            return holder

        def unpack(holder):
            session = _recompute_session.get()
            if session is not None:
                recompute_key = session.session_id
                retain_on_unpack = session.retain_on_unpack
            else:
                retain_on_unpack = False
                recompute_key = _pynative_executor.get_current_autodiff_engine_id()
                if recompute_key == -1:
                    recompute_key = int(uuid.uuid4())

            handle.recompute(recompute_key)
            if handle.recomputed_data[recompute_key].get(holder, None) is None:
                raise RuntimeError("Unpack is being triggered for a tensor, make sure to do this only once!")
            val = handle.recomputed_data[recompute_key][holder]
            if not retain_on_unpack:
                handle.recomputed_data[recompute_key].pop(holder)
            return val
        super().__init__(pack, unpack)


class _RecomputationHook(saved_tensors_hooks):
    """
    Recompute hook for saved tensor which need get activation value.
    """
    def __init__(self, weak_handle, recompute_key):
        def pack(x):
            x = ops.stop_gradient(x) if x._requires_grad else x  # pylint: disable=protected-access
            handle = weak_handle()
            if handle is None:
                raise RuntimeError("Recompute handle has been garbage collected unexpectedly.")
            current_idx = handle.placeholder_counter[recompute_key]
            handle.placeholder_counter[recompute_key] += 1

            if current_idx >= len(handle.placeholders):
                if not handle.early_stop and not handle.forward_completed:
                    return x
                raise RuntimeError('This forward function contains non-determinism and is non-reentrant, '
                                   'which cannot be recomputed.')

            placeholder = handle.placeholders[current_idx]()
            if placeholder is not None:
                handle.recomputed_data[recompute_key][placeholder] = x

            # Early stop check: if we have recomputed all needed tensors, stop recomputation
            if handle.early_stop and handle.placeholder_counter[recompute_key] == len(handle.placeholders):
                raise _StopRecomputationError
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


def recompute(block, *args, use_reentrant=True, output_recompute=False, early_stop=True, context_fn=None, **kwargs):
    r"""
    This function is used to reduce memory, when run block, rather than
    storing the intermediate activation computed in forward pass, we will recompute it in backward pass.

    Note:
        Recompute function only supports a block which inherites from Cell object.

    Args:
        block (Cell): Block to be recomputed.
        args (tuple): Inputs for block object to run forward pass.

    Keyword Arguments:
        use_reentrant (bool, optional): This keyword is only valid in PyNative mode.
            If use_reentrant=True is set, we will implement recomputation through
            a custom bprop function, which does not support differentiation of complex types
            such as List/Tuple, If use_reentrant=False is set, we will use the saved_tensors_hook functionality
            to implement recomputation, which supports differentiation of tensors inside complex types.
            Default: ``True``.
        output_recompute (bool, optional): If output_recompute=True is set, we will implement recomputation by
            saved_tensors_hook functionality by default. The output of this cell or function will not be stored by
            subsequent operations for backward. When there are two adjacent cells both requiring recomputation (where
            the output of one cell serves as the input to the other), the recomputation of these two cells will be
            merged. In this case, the output activation values of the first cell will not be stored. If
            output_recompute=False, we will not merge adjacent cells. Default: ``False``.
        early_stop (bool, optional): This keyword is only valid in PyNative mode when use_reentrant=False.
            If ``True``, non-reentrant recompute stops recomputation as soon as it has computed all needed Tensors.
            This can reduce unnecessary computation when the forward function contains operations that don't save
            tensors at the end (e.g., clone, view operations). This argument is ignored if ``use_reentrant=True``.
            Note: Higher-order differentiation is only supported when ``early_stop=False``. Default: ``True``.
        context_fn (Callable, optional): A callable returning a tuple of two context managers.
            The first context manager is applied during the forward pass, and the second is applied during
            recomputation. This is useful for applying different settings (e.g., enabling/disabling gradient
            computation, mixed precision) during forward and recomputation phases.
            This argument is ignored if ``use_reentrant=True``. Default: ``None``, returns two null contexts.
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
        return ops.recompute_block(block, output_recompute)(*args, **kwargs)
    if output_recompute:
        use_reentrant = False
    _check_validation(block, use_reentrant)
    if use_reentrant:
        if context_fn is not None:
            raise ValueError("context_fn is only supported when use_reentrant=False.")
        return _RecomputeCell(block)(*args, **kwargs)
    if context_fn is None:
        context_fn = null_context_fn
    return recompute_without_reentrant(block, output_recompute, early_stop, context_fn, *args, **kwargs)


def recompute_without_reentrant(block, output_recompute, early_stop, context_fn, *args, **kwargs):  # pylint: disable=too-many-locals
    """
    Compute block by recompute function using saved tensors hook.
    :param block:
    :param output_recompute:
    :param early_stop:
    :param context_fn:
    :param args:
    :param kwargs:
    :return:
    """
    save_rng_state = kwargs.pop("save_rng_state", True)
    pre_rng_state = get_rng_state()
    amp_strategy = get_curr_amp_strategy()

    effective_early_stop = early_stop
    if _enable_recompute_early_stop is not None:
        effective_early_stop = _enable_recompute_early_stop

    forward_ctx, recompute_ctx = context_fn()

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
        with recompute_ctx:
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

    handle = _RecomputeHandle(recompute_function, early_stop=effective_early_stop)
    handle_collector = _recompute_handle_collector.get()
    if handle_collector is not None:
        handle_collector.append(handle)
    if _pynative_executor.is_saved_tensor_hook_active():
        fake_val = ops.zeros((0,))
        fake_val.requires_grad_()
        handle.input_node = _InputSaver.apply(fake_val, kwargs, *args)
    else:
        handle.wrap_original_inputs(*args, **kwargs)

    with _CreatePlaceholderHook(handle), forward_ctx:
        result = wrapper_block(*args, **kwargs)
    # Mark forward as completed for handling grad within recompute scenarios
    handle.forward_completed = True
    return result


def recompute_generator(block, use_reentrant=True, output_recompute=False, early_stop=True, context_fn=None):
    """
    generator of recompute object.
    :param output_recompute:
    :param use_reentrant:
    :param early_stop:
    :param context_fn:
    :param block:
    :return:
    """
    if use_reentrant:
        if context_fn is not None:
            raise ValueError("context_fn is only supported when use_reentrant=False.")
        return _RecomputeCell(block)

    if context_fn is None:
        context_fn = null_context_fn

    def create_recompute_func(*args, **kwargs):
        return recompute_without_reentrant(block, output_recompute, early_stop, context_fn, *args, **kwargs)
    return create_recompute_func


recompute_registry.register(recompute_generator)

__all__ = ['recompute', '_set_recompute_early_stop', '_recompute_session_ctx', '_clear_recompute_session',
           '_recompute_handle_collector_ctx']
__all__.sort()
