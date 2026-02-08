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
"""dryrun."""
import traceback
import os
from mindspore.common import Tensor
from mindspore import log as logger
from mindspore.common import dtype as mstype
from mindspore._checkparam import is_stub_tensor


class TraceBack():
    """
    traceback warning logs in dryrun mode
    """
    def __init__(self):
        self.stack_str_set = set()

    def inject(self, method):
        """
        inject warning logs in dryrun mode
        """
        def new_method(*args, **kwargs):
            stack_list = traceback.format_list(traceback.extract_stack())
            stack_str = "".join(stack_list)
            if "Parameter" not in stack_str and stack_str not in self.stack_str_set:
                self.stack_str_set.add(stack_str)
                logger.warning("In dryrun mode, you cannot obtain real tensor value, and the traceback is {%s}",
                               stack_list)
            return method(*args, **kwargs)
        return new_method


def no_inject_traceback_for_print(self):
    if is_stub_tensor(self):
        self = self.stub_sync()
    if self.dtype == mstype.type_none:
        return "Unknown Tensor type!"
    if self.has_init:
        self.init_data()
    return str(Tensor.asnumpy(self))


def set_simulation():
    """
    This interface is used to enable the dryrun function. The dryrun function is mainly used to simulate the actual
    operation of the large model. After it is enabled, the memory usage, compilation information, etc. can be simulated
    without occupying device card. In the PyNative mode, once it is enabled, if values are fetched from the device to
    the host, the Python call stack log will be printed to inform users that these values are inaccurate.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore.utils import dryrun
        >>> import numpy as np
        >>> dryrun.set_simulation()
        >>> print(os.environ.get('MS_SIMULATION_LEVEL'))
        1
    """
    os.environ["MS_SIMULATION_LEVEL"] = "1"
    obj = TraceBack()
    Tensor.asnumpy = obj.inject(Tensor.asnumpy)
    Tensor.__getitem__ = obj.inject(Tensor.__getitem__)
    Tensor.is_contiguous = obj.inject(Tensor.is_contiguous)
    Tensor.flush_from_cache = obj.inject(Tensor.flush_from_cache)
    Tensor.__str__ = no_inject_traceback_for_print
    Tensor.tolist = obj.inject(Tensor.tolist)
    Tensor.__int__ = obj.inject(Tensor.__int__)
    Tensor.__float__ = obj.inject(Tensor.__float__)


def mock(mock_val, *args):
    """
    In the network, if some `if` branches need to use the actual execution values and
    the virtual execution cannot obtain
    them, this interface can be used to return simulated values. During actual execution, the correct results can be
    obtained and the execution values can be returned.

    Args:
        mock_val (Union[Value, Tensor]): The value you want to return.
        args (Union[Value, function]): The content you want to mock, it can be values, functions and so on.

    Returns:
        If dryrun is enabled, mock_val will be returned; otherwise,
        the actual execution values of args will be returned.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore.utils import dryrun
        >>> import numpy as np
        >>> dryrun.set_simulation()
        >>> a = ms.Tensor(np.random.rand(3, 3).astype(np.float32))
        >>> if dryrun.mock(True, a[0, 0] > 0.5):
        ...     print("return mock_val: True.")
        return mock_val: True
        >>>
        >>> import mindspore as ms
        >>> from mindspore.utils import dryrun
        >>> import numpy as np
        >>> a = ms.Tensor(np.ones((3, 3)).astype(np.float32))
        >>> if dryrun.mock(False, a[0, 0] > 0.5):
        ...     print("return real execution: True.")
        return real execution: True.
        >>>
        >>> import mindspore as ms
        >>> from mindspore.utils import dryrun
        >>> import numpy as np
        >>> a = ms.Tensor(np.ones((3, 3)).astype(np.float32))
        >>> if dryrun.mock(False, (a > 0.5).any):
        ...     print("return real execution: True.")
        return real execution: True.
    """
    if os.environ.get('MS_SIMULATION_LEVEL'):
        return mock_val
    if len(args) == 1:
        if callable(args[0]):
            return args[0]()
        return args[0]
    return args[0](*args[1:])
