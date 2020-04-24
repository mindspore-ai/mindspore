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
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from ._comm_helper import Backend, _get_rank_helper, _get_size_helper, \
    _get_world_rank_from_group_rank_helper, _get_group_rank_from_world_rank_helper, \
    _create_group_helper, _destroy_group_helper, HCCL_WORLD_COMM_GROUP, NCCL_WORLD_COMM_GROUP, \
    _get_local_rank_helper, _get_local_size_helper
from .._c_expression import init_hccl, finalize_hccl, init_gpu_collective


__all__ = ["init", "release", "get_rank", "get_local_rank", "get_group_size",
           "get_local_rank_size", "get_world_rank_from_group_rank",
           "get_group_rank_from_world_rank", "create_group", "destroy_group",
           "HCCL_WORLD_COMM_GROUP", "NCCL_WORLD_COMM_GROUP"]

DEFAULT_WORLD_COMM_GROUP = HCCL_WORLD_COMM_GROUP
DEFAULT_BACKEND = Backend("hccl")


def _get_group(group):
    """Get the global world group if the group is default world comm group."""
    if group == DEFAULT_WORLD_COMM_GROUP:
        return GlobalComm.WORLD_COMM_GROUP
    return group


class GlobalComm:
    """Global communication info."""
    BACKEND = DEFAULT_BACKEND
    WORLD_COMM_GROUP = DEFAULT_WORLD_COMM_GROUP


def init(backend_name="hccl"):
    """
    Init distributed backend, e.g., hccl/nccl, it is required before communication service can be used.

    Note:
        The full name of hccl is Huawei Collective Communication Library.
        The full name of nccl is NVIDIA Collective Communication Library.

    Args:
        backend_name (str): Backend.

    Raises:
        TypeError: If backend name is not a string.
        RuntimeError: If backend is invalid or distributed init fails.
    """
    if not isinstance(backend_name, str):
        raise TypeError("Backend name must be a string, but got {}".format(type(backend_name)))

    if backend_name == "hccl":
        init_hccl()
        GlobalComm.BACKEND = Backend("hccl")
        GlobalComm.WORLD_COMM_GROUP = HCCL_WORLD_COMM_GROUP
    elif backend_name == "nccl":
        init_gpu_collective()
        GlobalComm.BACKEND = Backend("nccl")
        GlobalComm.WORLD_COMM_GROUP = NCCL_WORLD_COMM_GROUP
    else:
        raise RuntimeError("Backend name {} is not supported.".format(backend_name))

    auto_parallel_context().set_communication_backend(backend_name)


def release():
    """
    Release distributed resource. e.g., hccl/nccl.

    Raises:
        RuntimeError: If distributed resource release fails.
    """
    finalize_hccl()


def get_rank(group=GlobalComm.WORLD_COMM_GROUP):
    """
    Gets rank ID for current device in specified collective communication group.

    Args:
        group (str): ProcessGroup, the process group to work on. Default: WORLD_COMM_GROUP.

    Returns:
        int, the rank ID of the calling process within the group.

    Raises:
        TypeError: If group is not a string.
        ValueError: If backend is invalid.
        RuntimeError: If hccl/nccl is not available or nccl not supports.
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
        RuntimeError: If hccl/nccl is not available or nccl not supports.
    """
    return _get_local_rank_helper(group=_get_group(group), backend=GlobalComm.BACKEND)


def get_group_size(group=GlobalComm.WORLD_COMM_GROUP):
    """
    Gets rank size of the specified collective communication group.

    Args:
        group (str): ProcessGroup, the process group to work on.

    Returns:
        int, the rank size of the group.

    Raises:
        TypeError: If group is not a string.
        ValueError: If backend is invalid.
        RuntimeError: If hccl/nccl is not available or nccl not supports.
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
        int, the local rank size where the calling process is being within the group.

    Raises:
        TypeError: If group is not a string.
        ValueError: If backend is invalid.
        RuntimeError: If hccl/nccl is not available or nccl not supports.
    """
    return _get_local_size_helper(group=_get_group(group), backend=GlobalComm.BACKEND)


def get_world_rank_from_group_rank(group, group_rank_id):
    """
    Gets the rank ID in world communication group corresponding to the rank ID in specified user communication group.

    Note:
        Nccl is not supported.
        The parameter group should not be "hccl_world_group".

    Args:
        group (str): The user communication group.
        group_rank_id (int): A rank ID in user communication group.

    Returns:
        int, the rank ID in world communication group.

    Raises:
        TypeError: If group_rank_id is not a int or group is not a string.
        ValueError: If group is 'hccl_world_group' or backend is invalid.
        RuntimeError: If hccl/nccl is not available or nccl not supports.
    """
    return _get_world_rank_from_group_rank_helper(group=group, group_rank_id=group_rank_id, backend=GlobalComm.BACKEND)


def get_group_rank_from_world_rank(world_rank_id, group):
    """
    Gets the rank ID in specified user communication group corresponding to the rank ID in world communication group.

    Note:
        Nccl is not supported.
        The parameter group should not be "hccl_world_group".

    Args:
        world_rank_id (int): A rank ID in world communication group.
        group (str): The user communication group.

    Returns:
        int, the rank ID in user communication group.

    Raises:
        TypeError: If world_rank_id is not a int or group is not a string.
        ValueError: If group is 'hccl_world_group' or backend is invalid.
        RuntimeError: If hccl/nccl is not available or nccl not supports.
    """
    return _get_group_rank_from_world_rank_helper(world_rank_id=world_rank_id, group=group, backend=GlobalComm.BACKEND)


def create_group(group, rank_ids):
    """
    Creates user collective communication group.

    Note:
        Nccl is not supported.
        The size of rank_ids should be larger than 1.
        Rank_ids should not have duplicate data.

    Args:
        group (str): ProcessGroup, the process group to create.
        rank_ids (list): List of device ID.

    Raises:
        TypeError: If group is not a string or rank_ids is not a list.
        ValueError: If rank_ids size is not larger than 1 or rank_ids has duplicate data or backend is invalid.
        RuntimeError: If hccl/nccl is not available or nccl not supports.
    Examples:
        >>> group = "0-1"
        >>> rank_ids = [0,1]
        >>> create_group(group, rank_ids)
    """
    _create_group_helper(group, rank_ids, backend=GlobalComm.BACKEND)


def destroy_group(group):
    """
    Destroys user collective communication group.

    Note:
        Nccl is not supported.
        The parameter group should not be "hccl_world_group".

    Args:
        group (str): ProcessGroup, the process group to destroy.

    Raises:
        TypeError: If group is not a string.
        ValueError: If group is "hccl_world_group" or backend is invalid.
        RuntimeError: If hccl/nccl is not available or nccl not supports.
    """
    _destroy_group_helper(group, backend=GlobalComm.BACKEND)
