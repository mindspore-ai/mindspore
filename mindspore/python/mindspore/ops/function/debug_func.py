# Copyright 2022 Huawei Technologies Co., Ltd
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

"""Operators for debug function."""

from mindspore.ops.operations.debug_ops import Print
from .._primitive_cache import _get_cache_prim


def print_(*input_x):
    """
    Outputs the inputs to stdout. The outputs are printed to screen by default.
    It can also be saved in a file by setting the parameter  `print_file_path` in `context`.
    Once set, the output will be saved in the file specified by print_file_path.
    :func:`mindspore.parse_print` can be employed to reload the data.
    For more information, please refer to :func:`mindspore.set_context` and :func:`mindspore.parse_print`.

    Note:
        In pynative mode, please use python print function.
        In Ascend platform with graph mode, the bool, int and float would be converted into Tensor to print, and
        str remains unchanged.
        This function is used for debugging. When too much data is printed at the same time,
        in order not to affect the main process, the framework may discard some data. If you need to record the
        data completely, you are recommended to use the `Summary` function, and can check
        `Summary <https://www.mindspore.cn/mindinsight/docs/en/r1.9/summary_record.html?highlight=summary#>`_.

    Args:
        input_x (Union[Tensor, bool, int, float, str]): The inputs of print_.
            Supports multiple inputs which are separated by ','.

    Returns:
        Invalid value, should be ignored.

    Raises:
        TypeError: If `input_x` is not one of the following: Tensor, bool, int, float, str.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.ones([2, 1]).astype(np.int32))
        >>> y = Tensor(np.ones([2, 2]).astype(np.int32))
        >>> result = ops.print_('Print Tensor x and Tensor y:', x, y)
        Print Tensor x and Tensor y:
        Tensor(shape=[2, 1], dtype=Int32, value=
        [[1]
         [1]])
        Tensor(shape=[2, 2], dtype=Int32, value=
        [[1 1]
         [1 1]])
    """
    print_op = _get_cache_prim(Print)()
    return print_op(*input_x)


__all__ = ['print_']

__all__.sort()
