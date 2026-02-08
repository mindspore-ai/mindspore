
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""dlpack for tensor."""
from mindspore._c_expression import TensorPy as TensorPy_


def from_dlpack(dlpack):
    r"""
    Converts a DLPack object to a MindSpore Tensor.

    This function allows for the sharing of tensor data from other deep learning frameworks that support DLPack.
    The data is not copied and the returned MindSpore Tensor shares the memory with the source tensor.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        dlpack (PyCapsule): The DLPack object to be converted, which is a capsule containing a pointer to a
            `DLManagedTensor`.

    Returns:
        Tensor, the MindSpore Tensor that shares memory with the DLPack object.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore.utils.dlpack import to_dlpack, from_dlpack
        >>> import numpy as np
        >>> # Create a MindSpore Tensor and convert it to DLPack
        >>> x = ms.Tensor(np.random.rand(2, 3), ms.float32)
        >>> dlpack_obj = to_dlpack(x)
        >>>
        >>> # Convert the DLPack object back to a MindSpore Tensor
        >>> y = from_dlpack(dlpack_obj)
        >>> print(x.shape == y.shape)
        True
    """
    return TensorPy_.from_dlpack(dlpack)


def to_dlpack(tensor):
    r"""
    Converts a MindSpore Tensor to a DLPack object.

    The DLPack format is a standard for sharing tensor data between different deep learning frameworks.
    The returned DLPack object is a Python capsule that can be consumed by other libraries that support DLPack.
    The capsule contains a pointer to a `DLManagedTensor` structure. The consumer of the DLPack object is responsible
    for releasing the memory.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        tensor (Tensor): The MindSpore Tensor to be converted.

    Returns:
        PyCapsule, a DLPack object that can be consumed by other libraries.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore.utils.dlpack import to_dlpack, from_dlpack
        >>> import numpy as np
        >>> # Convert a MindSpore Tensor to DLPack
        >>> x = ms.Tensor(np.random.rand(2, 3), ms.float32)
        >>> dlpack_obj = to_dlpack(x)
        >>>
        >>> # At this point, dlpack_obj can be used by other frameworks that support DLPack.
        >>> # For demonstration, we convert it back to a MindSpore Tensor.
        >>> y = from_dlpack(dlpack_obj)
        >>> print(x.shape == y.shape)
        True
    """
    if tensor.has_init:
        tensor.init_data()
    return TensorPy_.to_dlpack(tensor)
