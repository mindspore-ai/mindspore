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

__all__ = ['print_', 'tensordump']

from mindspore.ops.operations.debug_ops import Print
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from .._primitive_cache import _get_cache_prim


def print_(*input_x):
    r"""
    Outputs the inputs to stdout.
    The outputs are printed to screen by default.
    It can also be saved in a file by setting the parameter  `print_file_path` in `context`.
    For more information, please refer to :func:`mindspore.set_context`.
    In Ascend platform with graph mode, the environment variables `MS_DUMP_SLICE_SIZE` and `MS_DUMP_WAIT_TIME`
    can be set to solve operator execution failure when outputting big tensor or outputting tensor intensively.

    Note:
        In pynative mode, please use python print function.
        In Ascend platform with graph mode, the bool, int and float would be converted into tensor to print, and
        str remains unchanged.
        This function is used for debugging.

    Args:
        input_x (Union[Tensor, bool, int, float, str, tuple, list]): The inputs of print\_.
            Supports multiple inputs which are separated by ','.

    Returns:
        Invalid value, should be ignored.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor(mindspore.ops.ones([2, 1], mindspore.int32))
        >>> y = mindspore.tensor(mindspore.ops.ones([2, 2], mindspore.int32))
        >>> result = mindspore.ops.print_('Print Tensor x and Tensor y:', x, y)
        Print Tensor x and Tensor y:
        Tensor(shape=[2, 1], dtype=Int32, value=
        [[1],
         [1]])
        Tensor(shape=[2, 2], dtype=Int32, value=
        [[1, 1],
         [1, 1]])
    """
    print_op = _get_cache_prim(Print)()
    return print_op(*input_x)



def tensordump(file_name, tensor, mode='out'):
    """
    Save tensor in npy format.

    .. warning::
        - The parameter `mode` will no longer support the value 'all'.

    In Parallel situation, tensordump will dump slice of data at each rank.
    In Ascend platform with graph mode, Your code OpA --> OpB may compiled as OpA --> RedistributionOps --> OpB.

    Note: The redistribution operator is introduced,
    Due to inter-device communication and shard strategies in the static graph parallel scenario.

    In case of OpA --> OpB, the dump data of OpA's output is equal to OpB's input.

    But in case of OpA --> RedistributionOps --> OpB,
    The dump data of OpA's output is not equal to OpB's input (Due to the redistribution operators).
    So the parameter mode is to handle this situation.

    Assuming OpA's output is used as both tensordump's input parameter and OpB's input parameter.
    Different requirements of saving dump data can be achieved by configuring parameter `mode` :

    - If the `mode` is 'out', the dump data contains only OpA's output slice.
    - If the `mode` is 'in', the dump data contains only OpB's input slice.

    For `mode` 'in', the input slice npy file format is: fileName_dumpMode_dtype_id.npy.

    For `mode` 'out', the output slice npy file format is: fileName_dtype_id.npy.

    - fileName: Value of the parameter file_name
      (if parameter `file_name` is a user-specified path, the value of fileName is the last level of the path).
    - dumpMode: Value of the parameter `mode`.
    - dtype: The original data type. Data of type bfloat16 stored in the .npy file will be converted to float32.
    - id: An auto increment ID.

    Note:
        - In Ascend platform with graph mode, the environment variables `MS_DUMP_SLICE_SIZE` and `MS_DUMP_WAIT_TIME`
          can be set to solve operator execution failure when outputting big tensor or outputting tensor intensively.
        - The operator of tensordump doesn't support in control flow.
        - If current parallel mode is STAND_ALONE, `mode` should only be 'out'.
        - This function is used for debugging.

    Args:
        file_name (str): The path of the npy file saves.
        tensor (Tensor): The tensor that user want to dump.
        mode (str, optional): Used to control tensordump behavior, available value is one of ['in', 'out'].
            Default ``out`` .

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Using msrun command to run below example: msrun --worker_num=2 --local_worker_num=2
            --master_port=11450 --log_dir=msrun_log --join=True --cluster_time_out=300 tensordump_example.py

        >>> import os
        >>> import time
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import nn, context
        >>> from mindspore.communication import init, get_rank
        >>> from mindspore.parallel.auto_parallel import AutoParallel
        >>> from mindspore.nn.utils import no_init_parameters
        >>> init()
        >>> rank_id = get_rank()
        >>> dump_path = f'rank_{rank_id}_mul1_mul2.npy'
        >>> class Net(nn.Cell):
        ...     def __init__(self, strategy1, strategy2):
        ...         super(Net, self).__init__()
        ...         self.matmul1 = mindspore.ops.MatMul().shard(strategy1)
        ...         self.matmul2 = mindspore.ops.MatMul().shard(strategy2)
        ...
        ...     def construct(self, x, y, b):
        ...         out1 = self.matmul1(x, y)
        ...         mindspore.ops.tensordump(dump_path, out1, 'out')
        ...         out2 = self.matmul2(out1, b)
        ...         return out2
        ...
        >>> mindspore.set_context(mode=mindspore.GRAPH_MODE)
        >>> os.environ["MS_DEV_SAVE_GRAPHS"] = "2"
        >>> strategy1 = ((1, 2), (2, 1))
        >>> strategy2 = ((1, 2), (2, 1))
        >>> with no_init_parameters():
        ...     net = Net(strategy1, strategy2)
        >>> x = mindspore.tensor(0.1 * mindspore.ops.randn(64, 64), mindspore.float32)
        >>> y = mindspore.tensor(0.1 * mindspore.ops.randn(64, 64), mindspore.float32)
        >>> b = mindspore.tensor(0.1 * mindspore.ops.randn(64, 64), mindspore.float32)
        >>> parallel_net = AutoParallel(net, parallel_mode="semi_auto")
        >>> parallel_net.dataset_strategy(config="full_batch")
        >>> out = parallel_net(x, y, b)
        >>> print(f"out shape is: {out.shape}")
        out shape is (64, 64)
        >>> time.sleep(0.5) # npy file is generated asynchronously, spend an interval time then load it.
        >>> matmul1_output_slice = np.load(f'rank_{rank_id}_mul1_mul2_float32_0.npy')      # load matmul1's output slice
        >>> print(f"matmul1_output_slice is loaded, shape is: {matmul1_output_slice.shape}")
        matmul1_output_slice is loaded, shape is: (64, 64)
    """

    if not isinstance(file_name, str):
        raise TypeError(f"Parameter file_name should only be build_in str type but got: {type(file_name)}")
    if not isinstance(tensor, Tensor):
        raise TypeError(f"Parameter tensor should only be Tensor type, but got: {type(tensor)}")
    if not isinstance(mode, str):
        raise TypeError(f"Parameter mode should only be build_in str type, but got: {type(mode)}")
    mode_list = ['out', 'in']
    if mode not in mode_list:
        if mode == 'all':
            raise ValueError(f"Argument [mode] has been not supported value of 'all'.")
        raise ValueError(f"Parameter mode should in {mode_list}, but got {mode}")
    _tensordump = _get_cache_prim(P.TensorDump)(input_output=mode)
    return _tensordump(file_name, tensor)


__all__.sort()
