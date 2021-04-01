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
from mindspore import context, log as logger
from mindspore.context import ParallelMode
from mindspore._c_expression import reset_op_id
from mindspore.common.tensor import Tensor
from mindspore.common.dtype import dtype_to_nptype
from mindspore.common import dtype as mstype
from mindspore.communication.management import get_group_size, get_rank
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.common.seed import get_seed


def _get_parallel_mode():
    """Get parallel mode."""
    return auto_parallel_context().get_parallel_mode()


def _get_full_batch():
    """Get whether to use full_batch."""
    return auto_parallel_context().get_full_batch()

def _get_pipeline_stages():
    """Get pipeline stages"""
    return auto_parallel_context().get_pipeline_stages()

def _check_full_batch():
    """
    full_batch could only be used under semi_auto_parallel or auto_parallel, check it.

    Raises:
        RuntimeError: Using full_batch under neither semi_auto_parallel nor auto_parallel.
    """
    parallel_mode = _get_parallel_mode()
    full_batch = _get_full_batch()
    if ((parallel_mode not in ("semi_auto_parallel", "auto_parallel")) and full_batch):
        raise RuntimeError("full_batch could only be used under semi_auto_parallel or auto_parallel.")


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

def _get_gradients_mean():
    """Get if using gradients_mean."""
    return auto_parallel_context().get_gradients_mean()


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
    parameter_broadcast = auto_parallel_context().get_parameter_broadcast()

    if parallel_mode in ("data_parallel", "hybrid_parallel") and parameter_broadcast is False and get_seed() is None:
        logger.warning("You are suggested to use mindspore.context.set_auto_parallel_context(parameter_broadcast=True)"
                       " or mindspore.common.set_seed() to share parameters among multi-devices.")

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
    if op_path != "mindspore.ops.functional":
        op = cls(*arglist)
    else:
        op = cls
    op.set_prim_instance_name(instance_name)
    return op


def _reset_op_id():
    """Reset op id."""
    reset_op_id()


def _parallel_predict_check():
    """validate parallel model prediction"""
    if _get_parallel_mode() in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
        if not context.get_auto_parallel_context("full_batch"):
            raise RuntimeError('Model prediction only supports full batch dataset. Please set "full_batch" with True.')
        if context.get_auto_parallel_context("enable_parallel_optimizer"):
            raise RuntimeError('Model prediction does not support parallel optimizer. Please set'
                               '"enable_parallel_optimizer" with False.')


def _check_similar_layout(tensor_layout1, tensor_layout2):
    """check if two tensor layouts are same"""
    if tensor_layout1[1] != tensor_layout2[1]:
        return False
    for i in tensor_layout1[1]:
        if i == -1:
            continue
        if tensor_layout1[0][-1-i] != tensor_layout2[0][-1-i]:
            return False
    return True


def _check_same_layout(tensor_layout1, tensor_layout2):
    """check if two tensor layouts are same"""
    return tensor_layout1[0] == tensor_layout2[0] and tensor_layout1[1] == tensor_layout2[1]


def _remove_repeated_slices(tensor_layout):
    """generate unrepeated tensor layout"""
    import copy
    new_tensor_layout = copy.deepcopy(tensor_layout)
    dev_mat = tensor_layout[0][:]
    tensor_map = tensor_layout[1]
    for dim in range(len(dev_mat)):
        if dim not in tensor_map:
            dev_mat[-1-dim] = 1
    new_tensor_layout[0] = dev_mat
    return new_tensor_layout


def _infer_rank_list(train_map, predict_map=None):
    """infer checkpoint slices to be loaded"""
    ret = {}
    for param_name in train_map:
        train_layout = train_map[param_name]
        train_dev_mat = train_layout[0]
        dev_num = np.array(train_dev_mat).prod()
        new_train_layout = _remove_repeated_slices(train_layout)
        array = np.arange(dev_num).reshape(train_dev_mat)
        index = ()
        for i in new_train_layout[0]:
            if i == 1:
                index = index + (0,)
            else:
                index = index + (slice(None),)
        rank_list = array[index].flatten()
        if not predict_map:
            ret[param_name] = (rank_list, False)
            continue
        if param_name not in predict_map:
            logger.warning("predict_map does not contain %s", param_name)
            continue
        predict_layout = predict_map[param_name]
        dev_num = np.array(predict_layout[0]).prod()
        # optimization pass
        if _check_same_layout(train_layout, predict_layout):
            dev_rank = _get_global_rank()
            ret[param_name] = ([dev_rank], True)
            continue
        if _check_similar_layout(train_layout, predict_layout):
            if len(rank_list) == 1:
                ret[param_name] = (rank_list, True)
            elif len(rank_list) == dev_num:
                dev_rank = _get_global_rank()
                ret[param_name] = ([rank_list[dev_rank]], True)
            else:
                ret[param_name] = (rank_list, False)
        else:
            ret[param_name] = (rank_list, False)
    return ret
