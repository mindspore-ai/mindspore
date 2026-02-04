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
""" multiprocessiong reduce and restore function """

import os
import platform
import multiprocessing
from multiprocessing import reduction
try:
    import multiprocessing.resource_sharer
except ImportError:
    pass
from mindspore import mint
from mindspore.common import Tensor
from mindspore.common import UntypedStorage
from mindspore._c_expression import StoragePy


def reduce_tensor(tensor):
    """Serialize tensor."""
    if tensor._requires_grad and not tensor.is_leaf:  # pylint:disable=protected-access
        raise RuntimeError(
            "Tensor with requires grad or is leaf can not be reduced across process."
        )
    storage = tensor.untyped_storage()
    if storage.nbytes() == 0:
        metadata = (tensor.dtype, tensor.device, tensor._requires_grad,)  # pylint:disable=protected-access
        return (restore_tensor_empty, (type(tensor), metadata))
    metadata = (tensor.storage_offset(), tensor.shape, tensor.stride(), tensor._requires_grad,)  # pylint:disable=protected-access
    return (restore_tensor, (type(tensor), storage, metadata))


def restore_tensor(cls, storage, metadata):
    """Deserialize tensor."""
    storage_offset, shape, stride, requires_grad = metadata
    t = _restore_tensor(storage, storage_offset, shape, stride)
    t._requires_grad = requires_grad # pylint:disable=protected-access
    return t


def _restore_tensor(storage, storage_offset, shape, stride):
    """Deserialize tensor."""
    # create a tensor with the correct dtype/device
    t = mint.empty((0,), dtype=storage.dtype, device=storage.device)
    return t.set_(storage, storage_offset, shape, stride)


def restore_tensor_empty(cls, metadata):
    """Create empty tensor."""
    dtype, device, requires_grad = metadata
    t = mint.empty((0,), dtype=dtype, device=device)
    t._requires_grad = requires_grad  # pylint:disable=protected-access
    return t


def reduce_storage(storage):
    """Serialize storage."""
    fd, size, type_id = storage._share_fd_cpu_()  # pylint:disable=protected-access
    df = multiprocessing.reduction.DupFd(fd)
    metadata = (df, size, type_id)
    rebuild = restore_storage_fd
    return (rebuild, (type(storage),) + metadata)


def restore_storage_fd(cls, df, size, type_id):
    """Deserialize storage."""
    fd = df.detach()
    try:
        storage = cls._new_shared_fd_cpu(fd, size, type_id)  # pylint:disable=protected-access
        return storage
    finally:
        os.close(fd)


def init_reductions():
    """Register serialize and deserialize method."""
    if platform.system().lower() in {"windows", "darwin"}:
        return
    reduction.register(StoragePy, reduce_storage)
    reduction.register(UntypedStorage, reduce_storage)
    reduction.register(Tensor, reduce_tensor)
