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
"""Communication management API"""
from mindspore import context
from mindspore.parallel._ps_context import _is_role_pserver, _is_role_sched
from ._comm_helper import Backend, _get_rank_helper, _get_size_helper, \
    _get_world_rank_from_group_rank_helper, _get_group_rank_from_world_rank_helper, \
    _create_group_helper, _destroy_group_helper, HCCL_WORLD_COMM_GROUP, NCCL_WORLD_COMM_GROUP, \
    _get_local_rank_helper, _get_local_size_helper, GlobalComm
from .._c_expression import init_hccl, finalize_hccl, init_gpu_collective


__all__ = ["init", "release", "get_rank", "get_local_rank", "get_group_size",
           "get_local_rank_size", "get_world_rank_from_group_rank",
           "get_group_rank_from_world_rank", "create_group", "destroy_group",
           "HCCL_WORLD_COMM_GROUP", "NCCL_WORLD_COMM_GROUP"]

DEFAULT_WORLD_COMM_GROUP = HCCL_WORLD_COMM_GROUP

def _get_group(group):
    """Return the world communication group if the `group` is `DEFAULT_WORLD_COMM_GROUP`."""
    if group == DEFAULT_WORLD_COMM_GROUP:
        return GlobalComm.WORLD_COMM_GROUP
    return group



def init(backend_name=None):
    """
    Initialize distributed backend, e.g. HCCL/NCCL, it is required before using the communication service.

    Note:
        The full name of HCCL is Huawei Collective Communication Library.
        The full name of NCCL is NVIDIA Collective Communication Library.

    Args:
        backend_name (str): Backend.

    Raises:
        TypeError: If `backend_name` is not a string.
        RuntimeError: If device target is invalid, or backend is invalid, or distributed initialization fails.
    """
    if _is_role_pserver() or _is_role_sched():
        return
    device_target = context.get_context("device_target")
    if backend_name is None:
        if device_target == "Ascend":
            backend_name = "hccl"
        elif device_target == "GPU":
            backend_name = "nccl"
        else:
            raise RuntimeError("Device target {} is not supported.".format(device_target))
    if not isinstance(backend_name, str):
        raise TypeError("Backend name must be a string, but got {}".format(type(backend_name)))

    if backend_name == "hccl":
        if device_target != "Ascend":
            raise RuntimeError("Device target should be 'Ascend' to init hccl, but got {}".format(device_target))
        init_hccl()
        GlobalComm.BACKEND = Backend("hccl")
        GlobalComm.WORLD_COMM_GROUP = HCCL_WORLD_COMM_GROUP
        GlobalComm.INITED = True
    elif backend_name == "nccl":
        init_gpu_collective()
        GlobalComm.BACKEND = Backend("nccl")
        GlobalComm.WORLD_COMM_GROUP = NCCL_WORLD_COMM_GROUP
        GlobalComm.INITED = True
    else:
        raise RuntimeError("Backend name {} is not supported.".format(backend_name))

def release():
    """
    Release distributed resource. e.g. HCCL/NCCL.

    Raises:
        RuntimeError: If failed to release distributed resource.
    """
    finalize_hccl()


def get_rank(group=GlobalComm.WORLD_COMM_GROUP):
    """
    Get the rank ID for the current device in the specified collective communication group.

    Args:
        group (str): ProcessGroup, the process group to work on. Default: WORLD_COMM_GROUP.

    Returns:
        int, the rank ID of the calling process within the group.

    Raises:
        TypeError: If group is not a string.
        ValueError: If backend is invalid.
        RuntimeError: If HCCL/NCCL is not available or NCCL is not supported.
    """
    return _get_rank_helper(group=_get_group(group), backend=GlobalComm.BACKEND)


def get_local_rank(group=GlobalComm.WORLD_COMM_GROUP):
    """
    Gets local rank ID for current device in specified collective communication group.

    Note:
        Nccl is not supported.

    Args:
        group (str): ProcessGroup, the process group to work on. Default: WORLD_COMM_GROUP.

    Returns:
        int, the local rank ID of the calling process within the group.

    Raises:
        TypeError: If group is not a string.
        ValueError: If backend is invalid.
        RuntimeError: If HCCL/NCCL is not available or NCCL is not supported.
    """
    return _get_local_rank_helper(group=_get_group(group), backend=GlobalComm.BACKEND)


def get_group_size(group=GlobalComm.WORLD_COMM_GROUP):
    """
    Get the rank size of the specified collective communication group.

    Args:
        group (str): ProcessGroup, the process group to work on.

    Returns:
        int, the rank size of the group.

    Raises:
        TypeError: If group is not a string.
        ValueError: If backend is invalid.
        RuntimeError: If HCCL/NCCL is not available or NCCL is not supported.
    """
    return _get_size_helper(group=_get_group(group), backend=GlobalComm.BACKEND)


def get_local_rank_size(group=GlobalComm.WORLD_COMM_GROUP):
    """
    Gets local rank size of the specified collective communication group.

    Note:
        Nccl is not supported.

    Args:
        group (str): ProcessGroup, the process group to work on.

    Returns:
        int, the local rank size where the calling process is within the group.

    Raises:
        TypeError: If group is not a string.
        ValueError: If backend is invalid.
        RuntimeError: If HCCL/NCCL is not available or NCCL is not supported.
    """
    return _get_local_size_helper(group=_get_group(group), backend=GlobalComm.BACKEND)


def get_world_rank_from_group_rank(group, group_rank_id):
    """
    Get the rank ID in the world communication group corresponding to
    the rank ID in the specified user communication group.

    Note:
        NCCL is not supported.
        The parameter group should not be "hccl_world_group".

    Args:
        group (str): The user communication group.
        group_rank_id (int): A rank ID in user communication group.

    Returns:
        int, the rank ID in world communication group.

    Raises:
        TypeError: If `group_rank_id` is not an integer or the group is not a string.
        ValueError: If group is 'hccl_world_group' or backend is invalid.
        RuntimeError: If HCCL/NCCL is not available or NCCL is not supported.
    """
    return _get_world_rank_from_group_rank_helper(group=group, group_rank_id=group_rank_id, backend=GlobalComm.BACKEND)


def get_group_rank_from_world_rank(world_rank_id, group):
    """
    Get the rank ID in the specified user communication group corresponding to
    the rank ID in the world communication group.

    Note:
        NCCL is not supported.
        The parameter group should not be "hccl_world_group".

    Args:
        world_rank_id (int): A rank ID in the world communication group.
        group (str): The user communication group.

    Returns:
        int, the rank ID in the user communication group.

    Raises:
        TypeError: If world_rank_id is not an integer or the group is not a string.
        ValueError: If group is 'hccl_world_group' or backend is invalid.
        RuntimeError: If HCCL/NCCL is not available or NCCL is not supported.
    """
    return _get_group_rank_from_world_rank_helper(world_rank_id=world_rank_id, group=group, backend=GlobalComm.BACKEND)


def create_group(group, rank_ids):
    """
    Create a user collective communication group.

    Note:
        NCCL is not supported.
        The size of rank_ids should be larger than 1.
        Rank_ids should not have duplicate data.

    Args:
        group (str): ProcessGroup, the process group to create.
        rank_ids (list): A list of device IDs.

    Raises:
        TypeError: If group is not a string or `rank_ids` is not a list.
        ValueError: If `rank_ids` size is not larger than 1, or `rank_ids` has duplicate data, or backend is invalid.
        RuntimeError: If hccl/nccl is not available or nccl not supports.
    Examples:
        >>> init()
        >>> group = "0-1"
        >>> rank_ids = [0,1]
        >>> create_group(group, rank_ids)
    """
    _create_group_helper(group, rank_ids, backend=GlobalComm.BACKEND)


def destroy_group(group):
    """
    Destroy the user collective communication group.

    Note:
        Nccl is not supported.
        The parameter group should not be "hccl_world_group".

    Args:
        group (str): ProcessGroup, the process group to destroy.

    Raises:
        TypeError: If group is not a string.
        ValueError: If group is "hccl_world_group" or backend is invalid.
        RuntimeError: If HCCL/NCCL is not available or NCCL is not supported.
    """
    _destroy_group_helper(group, backend=GlobalComm.BACKEND)
