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
"""comm_helper"""

from mindspore.parallel._ps_context import _is_role_pserver, _is_role_sched
from ._hccl_management import load_lib as hccl_load_lib

_HCCL_AVAILABLE = False
_NCCL_AVAILABLE = False
try:
    import mindspore._ms_mpi as mpi
    _NCCL_AVAILABLE = True
except ImportError:
    _NCCL_AVAILABLE = False


try:
    hccl_load_lib()
    _HCCL_AVAILABLE = True
except RuntimeError:
    _HCCL_AVAILABLE = False

if _HCCL_AVAILABLE:
    from . import _hccl_management as hccl
else:
    try:
        import hccl_test.manage.api as hccl
        _HCCL_AVAILABLE = True
    except ImportError:
        _HCCL_AVAILABLE = False


HCCL_WORLD_COMM_GROUP = "hccl_world_group"
NCCL_WORLD_COMM_GROUP = "nccl_world_group"

class Backend:
    """
    Class for available backends.

    Note:
        The backends' value should be string, e.g., "hccl".
        If backend is set to Backend.UNDEFINED, it will be seen as invaliad.

    Args:
        name (str): The name of backend.

    Raises:
        TypeError: If name is not a string.
        ValueError: If backend is invalid.

    Examples:
        >>> Backend("abc")
        >>> hccl = Backend("hccl")
    """
    UNDEFINED = "undefined"
    HCCL = "hccl"
    NCCL = "nccl"

    def __new__(cls, name):
        """Create instance object of Backend."""
        if not isinstance(name, str):
            raise TypeError("Backend name must be a string, but got {}".format(type(name)))
        value = getattr(Backend, name.upper(), Backend.UNDEFINED)
        if value == Backend.UNDEFINED:
            raise ValueError("Invalid backend: '{}'".format(name))
        return value

DEFAULT_BACKEND = Backend("hccl")

class GlobalComm:
    """World communication information."""
    BACKEND = DEFAULT_BACKEND
    WORLD_COMM_GROUP = HCCL_WORLD_COMM_GROUP
    INITED = False

def is_hccl_available():
    """
    Check hccl api is available.

    Returns:
        Boolean. Return whether hccl is available or not.
    """
    return _HCCL_AVAILABLE


def is_nccl_available():
    """
    Check nccl api is available.

    Returns:
        Boolean. Return whether nccl is available or not.
    """
    return _NCCL_AVAILABLE


def check_parameter_available(func):
    """
    Check parameter is available. If not available, raise Error.

    Args:
        func (Function): The function to be run.

    Raises:
        RuntimeError.

    Returns:
        Wrapper. If not available, raise Error.
    """
    def wrapper(*args, **kargs):
        if _is_role_pserver() or _is_role_sched():
            return func(*args, **kargs)
        if not GlobalComm.INITED:
            raise RuntimeError("Distributed Communication has not been inited")
        group = None
        if "group" in kargs.keys():
            group = kargs.get("group")
            if group is not None and not isinstance(group, str):
                raise TypeError("Group should be str or None, "
                                "but got group {}".format(type(group)))

        if "backend" in kargs.keys():
            backend = kargs.get("backend")
            if backend is Backend.HCCL and not is_hccl_available():
                raise RuntimeError("Distributed Communication doesn't have HCCL built in")
            if backend is Backend.NCCL and not is_nccl_available():
                raise RuntimeError("Distributed Communication doesn't have NCCL built in")

        if group is None:
            if backend is Backend.HCCL:
                group = HCCL_WORLD_COMM_GROUP
            elif backend is Backend.NCCL:
                group = NCCL_WORLD_COMM_GROUP
        return func(*args, **kargs)
    return wrapper


@check_parameter_available
def _get_rank_helper(group, backend):
    """
    The Helper to do get_rank_id.

    Args:
        group (str): The communication group.
        backend (str): The backend, like "hccl".

    Raises:
        ValueError: If backend is invalid.

    Returns:
        Integer. The local rank id of the calling process.
    """
    rank_id = None
    if _is_role_pserver() or _is_role_sched():
        rank_id = 0
        return rank_id
    if backend == Backend.HCCL:
        if group == HCCL_WORLD_COMM_GROUP:
            rank_id = hccl.get_rank_id()
        else:
            rank_id = hccl.get_rank_id(group)
    elif backend == Backend.NCCL:
        rank_id = mpi.get_rank_id(group)
    else:
        raise ValueError("Invalid backend: '{}'".format(backend))
    return rank_id


@check_parameter_available
def _get_local_rank_helper(group, backend):
    """
    The Helper to do get_local_rank_id.

    Args:
        group (str): The communication group.
        backend (str): The backend, like "hccl".

    Raises:
        ValueError: If backend is invalid.

    Returns:
        Integer. The local rank id of the calling process.
    """
    rank_id = None
    if backend == Backend.HCCL:
        if group == HCCL_WORLD_COMM_GROUP:
            rank_id = hccl.get_local_rank_id()
        else:
            rank_id = hccl.get_local_rank_id(group)
    elif backend == Backend.NCCL:
        raise RuntimeError("Nccl doesn't support get_local_rank_id now.")
    else:
        raise ValueError("Invalid backend: '{}'".format(backend))
    return rank_id


@check_parameter_available
def _get_size_helper(group, backend):
    """
    The Helper to do get_rank_size.

    Args:
        group (str): The communication group.
        backend (str): The backend, like "hccl".

    Raises:
        ValueError: If backend is invalid.

    Returns:
        Integer. The rank size of specified group.
    """
    size = None
    if _is_role_pserver() or _is_role_sched():
        size = 1
        return size
    if backend == Backend.HCCL:
        if group == HCCL_WORLD_COMM_GROUP:
            size = hccl.get_rank_size()
        else:
            size = hccl.get_rank_size(group)
    elif backend == Backend.NCCL:
        size = mpi.get_rank_size(group)
    else:
        raise ValueError("Invalid backend: '{}'".format(backend))
    return size


@check_parameter_available
def _get_local_size_helper(group, backend):
    """
    The Helper to do get_local_rank_size.

    Args:
        group (str): The communication group.
        backend (str): The backend, like "hccl".

    Raises:
        ValueError: If backend is invalid.

    Returns:
        Integer. The local rank size where the calling process is being within specified group.
    """
    size = None
    if backend == Backend.HCCL:
        if group == HCCL_WORLD_COMM_GROUP:
            size = hccl.get_local_rank_size()
        else:
            size = hccl.get_local_rank_size(group)
    elif backend == Backend.NCCL:
        raise RuntimeError("Nccl doesn't support get_local_rank_size now.")
    else:
        raise ValueError("Invalid backend: '{}'".format(backend))
    return size


@check_parameter_available
def _get_world_rank_from_group_rank_helper(group, group_rank_id, backend):
    """
    The Helper to do get_world_rank_from_group_rank.

    Args:
        group (str): The user communication group.
        group_rank_id (int): A rank id in user communication group.
        backend (str): The backend, like "hccl".

    Raises:
        TypeError: If group_rank_id is not int.
        ValueError: If group is "hccl_world_group" or backend is invalid.

    Returns:
        Integer. A rank id in world communication group.
    """
    world_rank_id = None
    if not isinstance(group_rank_id, int):
        raise TypeError("group_rank_id should be int, but got type {}".format(type(group_rank_id)))
    if backend == Backend.HCCL:
        if group == HCCL_WORLD_COMM_GROUP:
            raise ValueError("Group cannot be 'hccl_world_group'. ")
        world_rank_id = hccl.get_world_rank_from_group_rank(group, group_rank_id)
    elif backend == Backend.NCCL:
        raise RuntimeError("Nccl doesn't support get_world_rank_from_group_rank now.")
    else:
        raise ValueError("Invalid backend: '{}'".format(backend))
    return world_rank_id


@check_parameter_available
def _get_group_rank_from_world_rank_helper(world_rank_id, group, backend):
    """
    The Helper to do get_group_rank_from_world_rank.

    Args:
        world_rank_id (int): A rank id in world communication group.
        group (str): The user communication group.
        backend (str): The backend, like "hccl".

    Raises:
        TypeError: If world_rank_id is not int.
        ValueError: If group is 'hccl_world_group' or backend is invalid.

    Returns:
        Integer. A rank id in user communication group.
    """
    group_rank_id = None
    if not isinstance(world_rank_id, int):
        raise TypeError("world_rank_id should be int, but got type {}".format(type(world_rank_id)))
    if backend == Backend.HCCL:
        if group == HCCL_WORLD_COMM_GROUP:
            raise ValueError("Group cannot be 'hccl_world_group'. ")
        group_rank_id = hccl.get_group_rank_from_world_rank(world_rank_id, group)
    elif backend == Backend.NCCL:
        raise RuntimeError("Nccl doesn't support get_group_rank_from_world_rank now.")
    else:
        raise ValueError("Invalid backend: '{}'".format(backend))
    return group_rank_id


@check_parameter_available
def _create_group_helper(group, rank_ids, backend):
    """
    The Helper to do create_group.

    Args:
        group (str): The communication group.
        rank_ids (list): Rank ids in the group.
        backend (str): The backend, like "hccl".

    Raises:
        TypeError: If rank_ids is not a list.
        ValueError: If rank_ids size is not larger than 1 or rank_ids has duplicate data or backend is invalid.
    """
    if backend == Backend.HCCL:
        if not isinstance(rank_ids, list):
            raise TypeError("Rank_ids {} should be list".format(rank_ids))
        rank_size = len(rank_ids)
        if rank_size < 1:
            raise ValueError("Rank_ids size {} should be large than 0".format(rank_size))
        if len(rank_ids) - len(list(set(rank_ids))) > 0:
            raise ValueError("List rank_ids in Group {} has duplicate data!".format(group))
        hccl.create_group(group, rank_size, rank_ids)
    elif backend == Backend.NCCL:
        raise RuntimeError("Nccl doesn't support create_group now.")
    else:
        raise ValueError("Invalid backend: '{}'".format(backend))


@check_parameter_available
def _destroy_group_helper(group, backend):
    """
    The Helper to do destroy_group.

    Args:
        group (str): The user communication group.
        backend (str): The backend, like "hccl".

    Raises:
        ValueError: If group is "hccl_world_group" or backend is invalid.
    """
    if backend == Backend.HCCL:
        if group == HCCL_WORLD_COMM_GROUP:
            raise ValueError("The hccl_world_group does not support destruction.")
        hccl.destroy_group(group)
    elif backend == Backend.NCCL:
        raise RuntimeError("Nccl doesn't support destroy_group now.")
    else:
        raise ValueError("Invalid backend: '{}'".format(backend))
