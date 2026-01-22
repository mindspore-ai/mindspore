# Copyright 2025 Huawei Technologies Co., Ltd
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
""" Distributed data parallel wrapper. """
from __future__ import absolute_import

__all__ = ["DistributedDataParallel"]

import itertools
from contextlib import contextmanager
from typing import Optional
import mindspore.nn as nn
import mindspore.log as logger
from mindspore import Tensor, mint
from mindspore.common import dtype as mstype
from mindspore.mint.distributed import get_world_size
from mindspore.communication import GlobalComm
from mindspore.graph.api import _pynative_executor
from mindspore.mint.distributed import broadcast, get_global_rank
from mindspore.parallel.distributed.flatten_grad_buffer import FlattenGradBuffer
from mindspore._c_expression import Reducer, _find_unused_parameters


def get_data_parallel_group():
    """get default global data parallel group"""
    return GlobalComm.WORLD_COMM_GROUP


def get_data_parallel_world_size(group):
    """get group world size"""
    return get_world_size(group)


def _find_tensors(obj):
    if isinstance(obj, Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return itertools.chain.from_iterable(map(_find_tensors, obj))
    if isinstance(obj, dict):
        return itertools.chain.from_iterable(map(_find_tensors, obj.values()))

    return []


class DistributedDataParallel(nn.Cell):
    """
    DistributedDataParallel wrapper. DistributedDataParallel allocates contiguous memory buffer for gradients.
    Parameters' gradients will be combined into multiple buckets which are the unit to conduct all-reduce
    communication among data parallel group to overlap communication latency.

    .. warning::
        - The method is currently only supported in PyNative mode.
        - This is an experimental interface, may be changed or canceled in the future.

    Args:
        module (nn.Cell): the module to be wrapped with DDP.
        init_sync (bool, optional): whether to sync params from rank0 of process_group when init. Default: ``True``.
        process_group (str, optional): the comm group of data prallel. Default: ``None``.
        bucket_cap_mb (int, optional): size of bucket in MB, default is 25MB if not set. Default: ``None``.
        find_unused_parameters (bool, optional): whether to find unused params in the bucket. Default: ``False``.
        average_in_collective (bool, optional): True means allreduce sum within DP group firstly then scaling with
            dp size. Otherwise scaling local rank grad first and then allreduce sum. Default: ``False``.
        static_graph (bool, optional): Indicate whether it is a static network. When it is a static network, the
            parameter `find_unused_parameters` will be ignored, and unused parameters will be searched for in the
            first step. Bucket reconstruction will be performed in execution order before the second step to achieve
            better performance. Default: ``False``.
        reducer_mode (str, optional): the backend to be used, could be "CppReducer" for cpp backend or "PythonReducer"
            for Python backend. Default: ``"CppReducer"``.

    Returns:
        Model wrapped with DistributedDataParallel.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            - When enabling recomputation or gradient freezing, the model should be wrapped by
              `DistributedDataParallel` at the outermost layer.
            - Before running the following examples, you need to configure the communication environment variables.
              For Ascend devices, it is recommended to use the msrun startup method
              without any third-party or configuration file dependencies. For detailed information, refer to
              `msrun launch <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_ .

        >>> from mindspore.parallel.distributed import DistributedDataParallel
        >>> from mindspore.mint.optim import AdamW
        >>> from mindspore import Parameter, Tensor, ops, nn
        >>> import mindspore as ms
        >>> from mindspore.communication import init
        >>> from mindspore.mint.distributed.distributed import init_process_group
        >>> ms.set_context(mode=ms.PYNATIVE_MODE)
        >>> init_process_group()
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> net = DistributedDataParallel(module=net,
        ...                              bucket_cap_mb=None,
        ...                              average_in_collective=True,
        ...                              static_graph=True)
        >>> optimizer = AdamW(net.trainable_params(), 1e-4)
        >>> loss_fn = nn.CrossEntropyLoss()
        >>>
        >>> def forward_fn(data, target):
        ...     logits = net(data)
        ...     loss = loss_fn(logits, target)
        ...     return loss, logits
        >>>
        >>> grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)
        >>>
        >>> # Create the dataset taking MNIST as an example. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/mnist.py
        >>> dataset = create_dataset()
        >>> for epoch in range(1):
        ...     step = 0
        ...     for image, label in dataset:
        ...         (loss_value, _), grads = grad_fn(image, label)
        ...         optimizer(grads)
        ...         net.zero_grad()
        ...         step += 1
        ...         print("epoch: %s, step: %s, loss is %.15f" % (epoch, step, loss_value))
    """

    def __init__(self, module, init_sync=True, process_group=None, bucket_cap_mb: Optional[int] = None,
                 find_unused_parameters=False, average_in_collective: bool = False, static_graph=False,
                 reducer_mode="CppReducer"):
        super(DistributedDataParallel, self).__init__(auto_prefix=False)
        self.init_sync = init_sync
        self.bucket_cap_mb = bucket_cap_mb
        self.average_in_collective = average_in_collective
        self.grad_reduce_in_fp32 = False
        self.process_group = process_group if process_group else get_data_parallel_group()
        self.static_graph = static_graph
        self.find_unused_parameters = find_unused_parameters

        self.module = module
        self.param_to_buffer = {}
        self.has_buckets_grad_sync = False

        # default is 25MB for each buck
        if bucket_cap_mb is None:
            bucket_cap_mb = 25
        self.bucket_bytes_cap = int(bucket_cap_mb * 1024 * 1024)

        # grads sync with allreduce comm
        self.sync_enabled = True
        self.reducer_mode = reducer_mode # "CppReducer" or "PythonReducer"
        self.buffers = []
        self.has_mark_unused_param = False

        bucketed_params = []
        self.skipped_params = []
        for _, param in self.module.parameters_and_names():
            if not param.requires_grad:
                self.skipped_params.append(param)
                continue
            param.grad = None
            param.main_grad = None
            bucketed_params.append(param)
            if self.average_in_collective:
                # allreduce to add grads, then to scale grads with dp size
                self.gradient_scaling_factor = 1.0
            else:
                # scale grads with dp size locally, then allreduce to add grads
                data_parallel_world_size = get_data_parallel_world_size(self.process_group)
                self.gradient_scaling_factor = 1.0 / data_parallel_world_size
        self.bucketed_params = bucketed_params

        if self.reducer_mode == "CppReducer":
            self.reducer = Reducer(self.bucketed_params,
                                   self.process_group,
                                   bucket_cap_mb,
                                   self.grad_reduce_in_fp32,
                                   average_in_collective,
                                   static_graph,
                                   find_unused_parameters)
            if self.init_sync:
                self.broadcast_coalesced()
            return
        # allocate buffer for trained params
        self.buffers = self.allocate_buffers_for_parameters(
            self.bucketed_params,
            group=self.process_group,
            gradient_scaling_factor=self.gradient_scaling_factor,
        )
        if self.init_sync:
            self.broadcast_coalesced()

        # register hook for bucket grad reduce
        self._register_hook_for_params()

        # bucket rebuilding
        self.rebuilt_params_ = []
        self.buffer_iterations = 0
        self.has_bucket_rebuilt = False
        self.buffer_issued = 0
        self.triggered_once = False

    def _group_params_by_dtype(self, input_params):
        param_and_grad_dtype_to_params = {}
        # group all params by parameter's data type and their gradient's data type.
        for param in input_params:
            param_dtype = param.dtype
            grad_dtype = mstype.float32 if self.grad_reduce_in_fp32 else param.dtype
            if (param_dtype, grad_dtype) not in param_and_grad_dtype_to_params:
                param_and_grad_dtype_to_params[(param_dtype, grad_dtype)] = []
            param_and_grad_dtype_to_params[(param_dtype, grad_dtype)].append(param)
        return param_and_grad_dtype_to_params

    def allocate_buffers_for_parameters(self, input_params, group, gradient_scaling_factor):
        """allocate buffers for parameters in different dtype group."""
        param_and_grad_dtype_to_params = self._group_params_by_dtype(input_params)

        buffers = []
        # allocate buffer for each group separately
        for (param_dtype, grad_dtype,), params in param_and_grad_dtype_to_params.items():
            buffers.append(
                FlattenGradBuffer(
                    average_in_collective=self.average_in_collective,
                    param_dtype=param_dtype,
                    grad_dtype=grad_dtype,
                    params=params,
                    data_parallel_group=group,
                    bucket_size=self.bucket_bytes_cap,
                    gradient_scaling_factor=gradient_scaling_factor,
                    ddp_handle=self,
                )
            )
            for param in params:
                self.param_to_buffer[param] = buffers[-1]
        logger.debug("allocate buffers for parameters: %s", buffers)
        return buffers

    def final_grad_reduce(self):
        """trigger final grad reduction"""
        logger.debug("trigger ddp final grad reduce, %d, %d", self.static_graph, len(self.unused_param))
        if self._should_rebuild_buckets():
            for param in self.unused_param:
                self.rebuilt_params_.append(param)
        for buffer in self.buffers:
            buffer.final_grad_reduce()
            buffer.issued = 0
        self.buffer_issued = 0

    def _register_hook_for_params(self):
        """register backward hook for each params."""
        for param in self.module.get_parameters():
            if param.requires_grad:
                param.register_hook(self._make_param_hook(param))

    def _post_forward(self, output):
        """prepare for backward (e.g. find unused params) if needed"""
        if self.reducer_mode == "CppReducer":
            if _pynative_executor.grad_flag() and self.sync_enabled:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
        else:
            unused_param_idx = []
            if self.static_graph and not self.triggered_once:
                self.triggered_once = True
                self.find_unused_parameters = False
                unused_param_idx = _find_unused_parameters(list(_find_tensors(output)), self.bucketed_params)
            elif self.find_unused_parameters:
                unused_param_idx = _find_unused_parameters(list(_find_tensors(output)), self.bucketed_params)
            self.unused_param = [self.bucketed_params[idx] for idx in unused_param_idx]
            self.unused_param_name = [param.name for param in self.unused_param]
            self.has_mark_unused_param = False

    def _pre_forward(self):
        """pre-process of forward pass to allocate buffer for parameters."""
        if self.reducer_mode == "CppReducer":
            if _pynative_executor.grad_flag() and self.sync_enabled:
                self.reducer.prepare_for_forward()
                self.reducer.rebuild_buckets()
            return
        if self.rebuilt_params_ and self._should_rebuild_buckets():
            for i in self.rebuilt_params_:
                i.old_grad = i.grad

            self.buffers = self.allocate_buffers_for_parameters(
                self.rebuilt_params_,
                group=self.process_group,
                gradient_scaling_factor=self.gradient_scaling_factor,
            )
            for buffer in self.buffers:
                buffer.sync_enabled = self.sync_enabled

            for i in self.rebuilt_params_:
                i.grad.copy_(i.old_grad)
                i.old_grad = None

            logger.debug("register unused param: %s", self.rebuilt_params_)
            self.has_bucket_rebuilt = True
            self.rebuilt_params_ = []

    def construct(self, *inputs, **inputs_dict):
        """construct for DistributedDataParallel."""
        self._pre_forward()
        output = self.module(*inputs, **inputs_dict)
        self._post_forward(output)
        return output

    def zero_grad(self):
        """DPP will accumulate grads automatically, it will zero grads when call zero_grad() manually."""
        if self.reducer_mode == "CppReducer":
            self.reducer.zero_grad()
        else:
            for buffer in self.buffers:
                buffer.reset()

    def _enable_sync(self, enable):
        """enable grad buffer sync or not."""
        for buffer in self.buffers:
            buffer.sync_enabled = enable
        self.sync_enabled = enable

    @contextmanager
    def no_sync(self):
        """Context manager helper function. When enabled, no grad allreduce synchronization will be executed."""
        self._enable_sync(False)
        try:
            yield
        finally:
            self._enable_sync(True)

    def _should_rebuild_buckets(self):
        if self.static_graph and not self.has_bucket_rebuilt:
            return True
        return False

    def _make_param_hook(self, param):
        """make closure function as the param hook."""
        def param_hook(grad):
            if not self.has_mark_unused_param:
                for cur_param in self.unused_param:
                    buffer = self.param_to_buffer[cur_param]
                    logger.debug("register unused param: %s", cur_param)
                    buffer.register_grad_ready(cur_param)
                self.has_mark_unused_param = True
            elif param.name in self.unused_param_name:
                logger.debug("unused param already registered: %s", param)
                return param.grad

            logger.debug("register normal param: %s", param)
            buffer = self.param_to_buffer[param]
            param.grad.add_(grad)
            buffer.register_grad_ready(param)
            if self._should_rebuild_buckets():
                self.rebuilt_params_.append(param)
            return param.grad

        return param_hook

    def broadcast_coalesced(self):
        """broadcast params from rank 0"""
        if self.reducer_mode == "CppReducer":
            buckets = [[self.bucketed_params[idx] for idx in bucket] for bucket in self.reducer.bucket_indices]
        else:
            buckets = [bucket.params_list for buffer in self.buffers for bucket in buffer.buckets]
        if self.skipped_params:
            param_and_grad_dtype_to_params = self._group_params_by_dtype(self.skipped_params)
            for params_list in param_and_grad_dtype_to_params.values():
                buckets.append(params_list)

        def finish(rate_limiter):
            for _ in rate_limiter:
                handle, coalesced, params = rate_limiter.pop(0)
                handle.wait()
                ptr = 0
                for param in params:
                    param.view(-1).copy_(coalesced[ptr:ptr + param.numel()])
                    ptr += param.numel()

        rate_limiter = []
        for params in buckets:
            flat_tensors = [t.view(-1) for t in params]
            coalesced = mint.cat(flat_tensors)
            global_rank = get_global_rank(self.process_group, 0)
            handle = broadcast(coalesced, src=global_rank, group=self.process_group, async_op=True)
            rate_limiter.append((handle, coalesced, params))

            if len(rate_limiter) >= 2:
                finish(rate_limiter)
        finish(rate_limiter)
