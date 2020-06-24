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

from mindspore._c_expression import reset_op_id
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


PARAMETER_CLONED_INDEX = 0


class _CloneInfo():
    """
    The clone info of parameter.

    Attributes:
        be_cloned (bool): Whether the parameter is cloned.
        cloned (bool): Whether the parameter clone from other parameter.
        be_cloned_index (tuple): If the parameter is cloned, generate one index per clone.
        cloned_index (int): If the parameter clone from other parameter, it has a unique index.
    """
    def __init__(self):
        self.be_cloned = False
        self.cloned = False
        self.be_cloned_index = []
        self.cloned_index = None


def _set_clone_info(clone_from, clone_to):
    """
    Set the clone info.

    Args:
        clone_from (_CloneInfo): The clone info of be_cloned parameter.
        clone_to (_CloneInfo): The clone info of cloned parameter.
    """
    global PARAMETER_CLONED_INDEX
    clone_to.be_cloned = False
    clone_to.cloned = True
    clone_to.be_cloned_index = []
    clone_to.cloned_index = PARAMETER_CLONED_INDEX

    clone_from.be_cloned = True
    clone_from.be_cloned_index.append(PARAMETER_CLONED_INDEX)

    PARAMETER_CLONED_INDEX = PARAMETER_CLONED_INDEX + 1


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
