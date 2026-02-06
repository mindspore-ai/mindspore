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
""" Param and grad buffer, bucket implemenatrion. """
from __future__ import absolute_import

__all__ = ["Bucket", "FlattenGradBuffer"]

from enum import Enum
import numpy as np
from mindspore import mint, Tensor
from mindspore.common.initializer import Zero
from mindspore.communication.management import get_group_size
import mindspore.communication.comm_func as comm_func


class BufferType(Enum):
    PARAM = 0
    GRAD = 1


MEM_ALIGN_SIZE = 512
ALIGN_BYTES = 32
MIN_BUCKET_SIZE = int(1 * 1024 * 1024)
DEFAULT_BUCKET_SIZE = int(25 * 1024 * 1024)


class Bucket:
    """
    Bucket to track a subset of parameters and gradients in the buffer. Bucket records the parameters
    whose gradient has already been computed. It also provide functionality to synchronize gradients among
    data parallel group when all parameters' graidents have been computed.

    Args:
        average_in_collective (bool): Scaling grads before/after AllReduce, True: scaling after AllReduce.
        params (List(Parameters)): Parameters belongs to this bucket.
        grad_data (Tensor): A section of buffers' gradient data, coressponding to parameters in this bucket.
        offset (int): Start index in the buffer.
        numel_unpadded (int): Number of unpadded elements in bucket.
        data_parallel_group (str): Data parallel group name.
        data_parallel_world_size (int): Data parallel group size.
        gradient_scaling_factor (float): Work with average_in_collective, it is 1.0 when average_in_collective
        true else 1.0/dp
    """

    def __init__(self, average_in_collective, params, grad_data, offset, numel_unpadded, data_parallel_group,
                 data_parallel_world_size, gradient_scaling_factor):
        self.average_in_collective = average_in_collective
        self.params_list = params
        self.params = set(params)
        self.params_grad_ready = set()
        self.grad_data = grad_data
        self.grad_data_numel = self.grad_data.numel()
        self.offset = offset
        self.numel_unpadded = numel_unpadded
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = data_parallel_world_size
        self.gradient_scaling_factor = gradient_scaling_factor

        if self.data_parallel_world_size > 1:
            self.grad_reducer = comm_func.all_reduce

        self.reset()

    def inplace_reduce_dp(self, src):
        """conduct all-reduce/reduce-scatter on src tensor and inplace update result into target."""
        self.communication_result, self.communication_handle = self.grad_reducer(
            src, "sum", self.data_parallel_group, async_op=True
        )

    def reset(self):
        """reset bucket for the next iteration."""
        self.params_grad_ready = set()
        self.is_reduce_issued = False
        self.communication_handle = None
        self.communication_result = None

    def issue_grad_reduce(self):
        """issue grad reduce for the local grad data view."""
        if self.is_reduce_issued:
            raise RuntimeError("The bucket reduce is already issued")

        if self.gradient_scaling_factor != 1.0:
            self.grad_data.copy_(mint.mul(self.grad_data, self.gradient_scaling_factor))

        if self.data_parallel_world_size > 1:
            self.inplace_reduce_dp(self.grad_data)

        self.is_reduce_issued = True

    def final_grad_reduce(self):
        """finalize grad reduce for the local grad data view."""
        start_idx = 0
        end_idx = self.grad_data_numel
        target = self.grad_data[start_idx:end_idx]

        if not self.is_reduce_issued:
            raise RuntimeError(
                f"The bucket reduce has not been issued "
                f"with only {len(self.params_grad_ready)}/{len(self.params)} params ready"
            )

        if self.data_parallel_world_size > 1:
            self.communication_handle.wait()
            target.copy_(self.communication_result)
            self.communication_result = None
            if self.average_in_collective:
                target.copy_(mint.div(target, self.data_parallel_world_size))

    def register_grad_ready(self, param):
        """register grad ready and issue bucket grad reduce when the bucket is ready."""
        if param not in self.params:
            raise ValueError("The param to be registered is not in the bucket")

        if param in self.params_grad_ready:
            raise ValueError(f"The param {param} is already registered")

        self.params_grad_ready.add(param)
        if len(self.params_grad_ready) == len(self.params):
            self.issue_grad_reduce()
            return True

        return False

    def __repr__(self):
        return f"Bucket (offset={self.offset}, param_lens={len(self.params)})"


class FlattenGradBuffer:
    """
    Allocate contiguous memory buffer for given parameters and corresponding gradients. Breaking
    up parameters and gradients buffer into small buckets, which is the unit for all-reduce/reduce-scatter
    communication during back-propagation.

    Args:
        average_in_collective (bool): Scaling grads before/after AllReduce, True: scaling after AllReduce.
        param_dtype (mindspore.dtype): The parameters' datatype.
        grad_dtype (mindspore.dtype): The gradients' datatype.
        params (List(Parameters)): Parameters belongs to this buffer.
        data_parallel_group (str): Data parallel group name.
        bucket_size (int): Bucket size threshold used to partition bucekts.
        gradient_scaling_factor (float):
    """

    def __init__(self, average_in_collective, param_dtype, grad_dtype, params, data_parallel_group,
                 bucket_size, gradient_scaling_factor, ddp_handle):
        super(FlattenGradBuffer, self).__init__()
        self.param_dtype = param_dtype
        self.grad_dtype = grad_dtype
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = get_group_size(group=self.data_parallel_group)
        self.gradient_scaling_factor = gradient_scaling_factor
        self.average_in_collective = average_in_collective

        self.buckets = []
        self.param_index_map = {}
        self.param_to_bucket = {}
        self.sync_enabled = True
        self.issued = 0
        self.ddp_handle = ddp_handle

        buckets_metadata = self.calc_partition_metadata(bucket_size, params)
        self.instantiate_buckets(buckets_metadata, params)

    def calc_partition_metadata(self, bucket_size, params):
        """calc bucket partition metadata"""
        # helper func
        def _need_new_bucket(bucket_numel, bucket_id):
            target_bucket_size = bucket_size
            if bucket_id == 0 and bucket_size == DEFAULT_BUCKET_SIZE:
                target_bucket_size = MIN_BUCKET_SIZE
            return (
                bucket_size is not None
                and bucket_numel != 0
                and bucket_numel >= target_bucket_size
            )

        def _build_bucket():
            nonlocal buckets_metadata, bucket_start_index, bucket_params, bucket_id
            bucket_end_index = data_start_index
            buckets_metadata.append(
                (bucket_start_index, bucket_end_index, bucket_params)
            )
            bucket_start_index = bucket_end_index
            bucket_id = bucket_id + 1
            bucket_params = []

        param_data_list = []
        buckets_metadata = []
        data_start_index = 0
        data_end_index = 0
        bucket_id = 0
        bucket_start_index = 0
        bucket_params = []
        for param in params[::]:  # traverse from the beginning
            last_bucket_numel = data_start_index - bucket_start_index
            if _need_new_bucket(last_bucket_numel, bucket_id):
                _build_bucket()
            data_end_index = data_start_index + param.numel()
            bucket_params.append(param)
            param_data_list.append(param)
            self.param_index_map[param] = (data_start_index, data_end_index, bucket_id)
            data_start_index = data_end_index

        # add bucket for the last few params which do not reach the bucket_size threshold
        if data_start_index - bucket_start_index > 0:
            bucket_end_index = data_start_index
            buckets_metadata.append(
                (bucket_start_index, bucket_end_index, bucket_params)
            )
            data_start_index = bucket_end_index

        # allocate contiguous memory for parameters and gradients
        self.numel = data_start_index
        self.grad_data = Tensor(shape=(self.numel), dtype=self.grad_dtype, init=Zero())
        self.grad_data.init_data()
        self.numel_unpadded = 0
        return buckets_metadata

    def instantiate_buckets(self, buckets_metadata, params):
        """build bucket instance according to partition metadata"""
        for bucket_start_index, bucket_end_index, bucket_params in buckets_metadata:
            local_grad_data = self.grad_data[bucket_start_index:bucket_end_index]
            self.numel_unpadded += bucket_end_index - bucket_start_index
            bucket = Bucket(
                average_in_collective=self.average_in_collective,
                params=bucket_params,
                grad_data=local_grad_data,
                offset=bucket_start_index,
                numel_unpadded=bucket_end_index - bucket_start_index,
                data_parallel_group=self.data_parallel_group,
                data_parallel_world_size=self.data_parallel_world_size,
                gradient_scaling_factor=self.gradient_scaling_factor,
            )
            self.buckets.append(bucket)
            for param in bucket_params:
                self.param_to_bucket[param] = bucket

        for param in params:
            data_start_index, _, _ = self.param_index_map[param]
            param.grad = self._get_buffer_slice(
                param.shape, data_start_index, BufferType.GRAD
            )

    def _get_buffer_slice(self, shape, start_index, buffer_type):
        """get the buffer view with the same shape"""
        end_index = start_index + int(np.prod(shape))
        if start_index < 0 or end_index > self.numel:
            raise ValueError("index out of range")
        if buffer_type == BufferType.GRAD:
            buffer_tensor = self.grad_data[start_index:end_index]
        else:
            raise TypeError("Invalid buffer type for _get_buffer_slice.")
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor

    def reset(self):
        """reset buffer for the next iteration."""
        self.grad_data.zero_()
        for bucket in self.buckets:
            bucket.reset()
        self.sync_enabled = True

    def final_grad_reduce(self):
        """finalize grad reduce for each bucket"""
        for bucket in self.buckets:
            bucket.final_grad_reduce()

    def register_grad_ready(self, param):
        """register ready grad in its buckets"""
        if self.sync_enabled:
            bucket = self.param_to_bucket[param]
            if bucket.register_grad_ready(param):
                self.issued += 1
            if self.issued == len(self.buckets):
                self.ddp_handle.buffer_issued += 1
                if self.ddp_handle.buffer_issued == len(self.ddp_handle.buffers):
                    self.ddp_handle.final_grad_reduce()

    def __repr__(self):
        param_index_with_name = {
            param.name: index for (param, index) in self.param_index_map.items()
        }
        return f"Buffer has buckets: \n {self.buckets} \n and param_index_map: \n {param_index_with_name}"
