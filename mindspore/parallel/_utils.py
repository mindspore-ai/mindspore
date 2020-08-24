# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Utils of auto parallel"""

import numpy as np
from mindspore._c_expression import reset_op_id
from mindspore.common.tensor import Tensor
from mindspore.common.dtype import dtype_to_nptype
from mindspore.common import dtype as mstype
from mindspore.communication.management import get_group_size, get_rank
from mindspore.parallel._auto_parallel_context import auto_parallel_context


def _get_parallel_mode():
    """Get parallel mode."""
    return auto_parallel_context().get_parallel_mode()


def _get_full_batch():
    """Get whether to use full_batch."""
    return auto_parallel_context().get_full_batch()


def _need_to_full():
    """Check whether to convert input to full shape or tensor."""
    parallel_mode = _get_parallel_mode()
    full_batch = _get_full_batch()
    need = ((parallel_mode in ("semi_auto_parallel", "auto_parallel"))
            and (not full_batch))
    return need

def _to_full_shapes(shapes, device_num):
    """Expanding batch dimension according to device_num, adapt to mindspore minddata graph solution."""
    new_shapes = []
    for shape in shapes:
        new_shape = ()
        for i, item in enumerate(shape):
            if i == 0:
                new_shape += (item * device_num,)
            else:
                new_shape += (item,)
        new_shapes.append(new_shape)
    return new_shapes

def _to_full_tensor(elem, device_num, global_rank, scaling_sens=None):
    """Convert numpy to tensor, expanding batch dimension according to device_num, adapt to feed the data
       from host solution."""
    lst = []
    if not isinstance(elem, (tuple, list)):
        elem = [elem]
    if global_rank >= device_num:
        raise ValueError("The global rank must be smaller than device number, the global rank is {}, "
                         "the device num is {}".format(global_rank, device_num))

    for data in elem:
        if isinstance(data, np.ndarray):
            data = Tensor(data)
        if not isinstance(data, Tensor):
            raise ValueError("elements in tensors must be Tensor")
        shape_ = data.shape
        type_ = data.dtype
        new_shape = ()
        batchsize_per_device = 1
        for i, item in enumerate(shape_):
            if i == 0:
                new_shape += (item * device_num,)
                batchsize_per_device = item
            else:
                new_shape += (item,)
        new_tensor_numpy = np.zeros(new_shape, dtype_to_nptype(type_))
        start = global_rank * batchsize_per_device
        new_tensor_numpy[start: start + batchsize_per_device] = data.asnumpy()
        new_tensor = Tensor(new_tensor_numpy)
        lst.append(new_tensor)
    if scaling_sens:
        lst.append(Tensor(scaling_sens, mstype.float32))
    return tuple(lst)

def _get_mirror_mean():
    """Get if using mirror_mean."""
    return auto_parallel_context().get_mirror_mean()


def _get_device_num():
    """Get the device num."""
    parallel_mode = auto_parallel_context().get_parallel_mode()
    if parallel_mode == "stand_alone":
        device_num = 1
        return device_num

    if auto_parallel_context().get_device_num_is_set() is False:
        device_num = get_group_size()
    else:
        device_num = auto_parallel_context().get_device_num()
    return device_num


def _get_global_rank():
    """Get the global rank."""
    parallel_mode = auto_parallel_context().get_parallel_mode()
    if parallel_mode == "stand_alone":
        global_rank = 0
        return global_rank

    if auto_parallel_context().get_global_rank_is_set() is False:
        global_rank = get_rank()
    else:
        global_rank = auto_parallel_context().get_global_rank()
    return global_rank


def _get_parameter_broadcast():
    """Get the parameter broadcast."""
    parallel_mode = auto_parallel_context().get_parallel_mode()
    if parallel_mode == "stand_alone":
        parameter_broadcast = False
        return parameter_broadcast

    if auto_parallel_context().get_parameter_broadcast_is_set() is True:
        parameter_broadcast = auto_parallel_context().get_parameter_broadcast()
    elif parallel_mode in ("data_parallel", "hybrid_parallel"):
        parameter_broadcast = True
    else:
        parameter_broadcast = False

    return parameter_broadcast


def _device_number_check(parallel_mode, device_number):
    """
    Check device num.

    Args:
        parallel_mode (str): The parallel mode.
        device_number (int): The device number.
    """
    if parallel_mode == "stand_alone" and device_number != 1:
        raise ValueError("If parallel_mode is stand_alone, device_number must be 1, "
                         "device_number: {0}, parallel_mode:{1}".format(device_number, parallel_mode))


def _parameter_broadcast_check(parallel_mode, parameter_broadcast):
    """
    Check parameter broadcast.

    Note:
        If parallel mode is semi_auto_parallel or auto_parallel, parameter broadcast is not supported. Using the same
        random seed to make sure parameters on multiple devices are the same.

    Args:
        parallel_mode (str): The parallel mode.
        parameter_broadcast (bool): The parameter broadcast.

    Raises:
        ValueError: If parameter is broadcasted
                    but the parallel mode is "stand_alone" or "semi_auto_parallel" or "auto_parallel").
    """
    if parameter_broadcast is True and parallel_mode in ("stand_alone", "semi_auto_parallel", "auto_parallel"):
        raise ValueError("stand_alone, semi_auto_parallel and auto_parallel "
                         "do not support parameter broadcast, parallel_mode: {0}, parameter_broadcast:{1}"
                         .format(parallel_mode, parameter_broadcast))

def _get_python_op(op_name, op_path, instance_name, arglist):
    """Get python operator."""
    module = __import__(op_path, fromlist=["None"])
    cls = getattr(module, op_name)
    op = cls(*arglist)
    op.set_prim_instance_name(instance_name)
    return op


def _reset_op_id():
    """Reset op id."""
    reset_op_id()
