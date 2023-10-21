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

import os
import glob
import ctypes

import sys
from sys import excepthook

from mindspore import context
from mindspore.parallel._ps_context import _is_role_worker, _is_role_pserver, _is_role_sched, _is_ps_mode,\
                                           _get_ps_context
from mindspore import log as logger
from mindspore._c_expression import CollectiveManager, set_cluster_exit_with_exception, MSContext
from mindspore.common._utils import load_lib

HCCL_LIB = 'libhccl_plugin.so'


def hccl_load_lib():
    """load hccl lib"""
    try:
        base_dir = os.path.dirname(os.path.realpath(__file__))
        lib_path = os.path.join(base_dir, "../lib/plugin/ascend", HCCL_LIB)
        ctypes.CDLL(lib_path)
    except Exception as exc:
        raise RuntimeError('Get hccl lib error.') from exc

_HCCL_TEST_AVAILABLE = False

try:
    if MSContext.get_instance().is_ascend_plugin_loaded():
        hccl_load_lib()
except RuntimeError:
    _HCCL_TEST_AVAILABLE = True

if _HCCL_TEST_AVAILABLE:
    try:
        import hccl_test.manage.api as hccl
    except ImportError:
        _HCCL_TEST_AVAILABLE = False


HCCL_WORLD_COMM_GROUP = "hccl_world_group"
NCCL_WORLD_COMM_GROUP = "nccl_world_group"
MCCL_WORLD_COMM_GROUP = "mccl_world_group"

DEVICE_TO_BACKEND = {
    "Ascend": "hccl",
    "GPU": "nccl",
    "CPU": "mccl"
}

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
    MCCL = "mccl"

    @staticmethod
    def __new__(cls, name):
        """Create instance object of Backend."""
        if not isinstance(name, str):
            raise TypeError("For 'Backend', the class variable 'name' must be a string, "
                            "but got the type : {}".format(type(name)))
        value = getattr(Backend, name.upper(), Backend.UNDEFINED)
        if value == Backend.UNDEFINED:
            raise ValueError("For 'Backend', the class variable 'name' {} is not supported, "
                             "please use hccl or nccl.".format(name))
        return value


DEFAULT_BACKEND = Backend("hccl")


class GlobalComm:
    """
    World communication information. The GlobalComm is a global class. The members contain:

    - ``BACKEND`` : The communication library used, using ``"hccl"`` / ``"nccl"`` / ``"mccl"`` .
      ``"hccl"`` means Huawei Collective Communication Library(HCCL),
      ``"nccl"`` means NVIDIA Collective Communication Library(NCCL),
      ``"mccl"`` means MindSpore Collective Communication Library(MCCL).
    - ``WORLD_COMM_GROUP`` : Global communication domain,
      using ``"hccl_world_group"`` / ``"nccl_world_group"`` / ``"mccl_world_group"`` .
    """
    BACKEND = DEFAULT_BACKEND
    WORLD_COMM_GROUP = HCCL_WORLD_COMM_GROUP
    INITED = False
    CHECK_ENVS = True


class _ExistingGroup:
    """
    The communication groups which exist in the progress.
    """
    ITEMS = {}


def _hccl_test():
    return _HCCL_TEST_AVAILABLE and GlobalComm.BACKEND == Backend.HCCL


def _check_mpi_envs():
    """
    Check whether mpi environment variables have been exported or not.

    return True if mpi environment variables have been exported, False otherwise.
    """
    ompi_command_env = os.getenv("OMPI_COMMAND")
    pmix_rank_env = os.getenv("PMIX_RANK")
    if ompi_command_env and pmix_rank_env:
        return True
    return False


def _check_bypass_rank_id_and_size():
    '''
    Whether bypass calling c++ API to get rank id and size, instead, use fake rank id 0 and rank size 1.
    This returns True when this process is Scheduler node or is Server node in old Parameter Server training mode.
    '''
    if _is_role_sched():
        return True
    device_target = context.get_context("device_target")
    if _is_ps_mode() and _get_ps_context("worker_num") == 1 and device_target == "Ascend":
        return True
    return False


def _set_elegant_exit_handle():
    if _is_role_worker() or _is_role_pserver() or _is_role_sched():
        sys.excepthook = lambda *args: (set_cluster_exit_with_exception(), excepthook(*args))


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
        if not GlobalComm.INITED:
            raise RuntimeError("Distributed Communication has not been inited")
        group = None
        if "group" in kargs.keys():
            group = kargs.get("group")
            if group is not None and not isinstance(group, str):
                raise TypeError("The parameter 'group' should be str or None, "
                                "but got the type : {}".format(type(group)))
        if group is None:
            group = GlobalComm.WORLD_COMM_GROUP
        return func(*args, **kargs)
    return wrapper


def _is_available():
    """
    Returns `True` if distributed module is available.

    Note:
        Always returns `True` because MindSpore always has distributed ability on all platforms.
    """
    return True


def _is_initialized():
    """
    Checks if distributed module is successfully initialized.
    """
    return CollectiveManager.get_instance().initialized()


def _get_backend():
    """
    Returns the backend of communication process groups.

    Note:
        Only one communication backend is supported by MindSpore for each process.
        It should be one of `hccl`/`nccl`/`mccl`.
    """
    return GlobalComm.BACKEND


def _is_hccl_available():
    """
    Checks if `hccl` backend is available.
    """
    return _HCCL_TEST_AVAILABLE


def _is_nccl_available():
    """
    Checks if `nccl` backend is available.
    """
    base_dir = os.path.dirname(os.path.realpath(__file__))
    lib_path = os.path.join(base_dir, "../lib/plugin/gpu*/libnvidia_collective.so")
    file_paths = glob.glob(lib_path)
    return all(list(load_lib(f) for f in file_paths))


def _is_mpi_available():
    """
    Checks if OpenMPI's library is available.
    """
    base_dir = os.path.dirname(os.path.realpath(__file__))
    lib_path = os.path.join(base_dir, "../lib/libmpi_collective.so")
    return load_lib(lib_path)


@check_parameter_available
def _get_rank_helper(group):
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
    if _check_bypass_rank_id_and_size():
        rank_id = 0
        return rank_id
    if _hccl_test():
        return hccl.get_rank_id(group)
    rank_id = CollectiveManager.get_instance().get_rank_id(group)
    return rank_id


@check_parameter_available
def _get_local_rank_helper(group):
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
    rank_id = CollectiveManager.get_instance().get_local_rank_id(group)
    return rank_id


@check_parameter_available
def _get_size_helper(group):
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
    if _check_bypass_rank_id_and_size():
        size = 1
        return size
    if _hccl_test():
        return hccl.get_rank_size(group)
    size = CollectiveManager.get_instance().get_group_size(group)
    return size


@check_parameter_available
def _get_local_size_helper(group):
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
    size = CollectiveManager.get_instance().get_local_group_size(group)
    return size


@check_parameter_available
def _get_world_rank_from_group_rank_helper(group, group_rank_id):
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
    if not isinstance(group_rank_id, int):
        raise TypeError("For 'get_world_rank_from_group_rank', the argument 'group_rank_id' must be"
                        " type of int, but got 'group_rank_id' type : {}.".format(type(group_rank_id)))
    if _hccl_test():
        return hccl.get_world_rank_from_group_rank(group, group_rank_id)
    world_rank_id = CollectiveManager.get_instance().get_world_rank_from_group_rank(group, group_rank_id)
    return world_rank_id


@check_parameter_available
def _get_group_rank_from_world_rank_helper(world_rank_id, group):
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
        raise TypeError("For 'get_group_rank_from_world_rank', the argument 'world_rank_id' must be type of int, "
                        "but got 'world_rank_id' type : {}.".format(type(world_rank_id)))
    if _hccl_test():
        return hccl.get_group_rank_from_world_rank(world_rank_id, group)
    group_rank_id = CollectiveManager.get_instance().get_group_rank_from_world_rank(world_rank_id, group)
    return group_rank_id


@check_parameter_available
def _get_group_ranks(group):
    """
    The Helper to do get_group_ranks.

    Args:
        group (str): The communication group.

    Returns:
        List. The ranks of specified group.
    """
    return CollectiveManager.get_instance().get_group_ranks(group)


@check_parameter_available
def _create_group_helper(group, rank_ids):
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
    if group in _ExistingGroup.ITEMS.keys():
        if rank_ids != _ExistingGroup.ITEMS.get(group):
            raise ValueError("The group {} has been created, the rank_list is {}, "
                             "but current rank_list for the group is {}".
                             format(group, _ExistingGroup.ITEMS[group], rank_ids))
        logger.warning("%r group has existed.", group)
        return
    if not isinstance(rank_ids, list):
        raise TypeError("For 'create_group', the argument 'rank_ids' must be type of list, "
                        "but got 'rank_ids' type : {}.".format(type(rank_ids)))
    rank_size = len(rank_ids)
    if rank_size < 1:
        raise ValueError("For 'create_group', the argument 'rank_ids' size should be greater than 1, "
                         "but got 'rank_ids' size : {}.".format(len(rank_ids)))
    if len(rank_ids) - len(list(set(rank_ids))) > 0:
        raise ValueError("List rank_ids in Group {} has duplicate data!".format(group))
    if _hccl_test():
        hccl.create_group(group, rank_size, rank_ids)
    else:
        result = CollectiveManager.get_instance().create_group(group, rank_ids)
        if not result:
            raise RuntimeError("Failed to create communication group for {} with rank ids {}. "
                               "If NCCL is used, 'export NCCL_DEBUG=INFO' "
                               "is suggested before launching jobs.".format(group, rank_ids))

    _ExistingGroup.ITEMS[group] = rank_ids


@check_parameter_available
def _destroy_group_helper(group):
    """
    The Helper to do destroy_group.

    Args:
        group (str): The user communication group.
        backend (str): The backend, like "hccl".

    Raises:
        ValueError: If group is "hccl_world_group" or backend is invalid.
    """
    if group == GlobalComm.WORLD_COMM_GROUP:
        raise ValueError("The world_group does not support destruction.")
    if _hccl_test():
        hccl.create_group(group)
    else:
        CollectiveManager.get_instance().destroy_group(group)
