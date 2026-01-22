# Copyright 2024 Huawei Technologies Co., Ltd
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

"""JIT begin/end for various JIT context compile."""

from mindspore.graph.jit_trace import _jit_trace_begin, _jit_trace_end


def _jit_begin(fn_name, *args):
    """
    Start to build a MindIR func graph for a code snippet.

    This allows the MindSpore runtime to apply optimizations based on generated func graph.

    Note:
        Use it with `jit_end` cooperatively.

    Also see: :func:`jit_end`.

    Args:
        fn_name (str): The name of func graph to be built.
        args (tuple): The arguments of func graph.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.graph.jit_begin_end import _jit_begin as jit_begin
        >>> from mindspore.graph.jit_begin_end import _jit_end as jit_end
        ...
        >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        >>> def tensor_add(x, y):
        ...     jit_begin(x, y)
        ...     z = x + y
        ...     z = jit_end(z)
        ...     return z
        ...
        >>> out = tensor_add(x, y)
    """
    return _jit_trace_begin(fn_name, *args)


def _jit_end(*output_args):
    """
    Finish building a MindIR func graph for a code snippet.

    This allows the MindSpore runtime to apply optimizations based on generated func graph.

    Note:
        Use it with `jit_begin` cooperatively.

    Also see: :func:`jit_begin`.

    Args:
        output_args (tuple): The output of func graph.

    Returns:
        The same as args `output_args`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.graph.jit_begin_end import _jit_begin as jit_begin
        >>> from mindspore.graph.jit_begin_end import _jit_end as jit_end
        ...
        >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        >>> def tensor_add(x, y):
        ...     jit_begin(x, y)
        ...     z = x + y
        ...     z = jit_end(z)
        ...     return z
        ...
        >>> out = tensor_add(x, y)
    """
    return _jit_trace_end(*output_args)
