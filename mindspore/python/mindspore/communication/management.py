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
import os
from mindspore import context
from mindspore.parallel._ps_context import _is_ps_mode, _is_role_pserver, _is_role_sched, _get_ps_context
from mindspore.communication._comm_helper import Backend, _get_rank_helper, _get_size_helper, \
    _get_world_rank_from_group_rank_helper, _get_group_rank_from_world_rank_helper, \
    _create_group_helper, _destroy_group_helper, HCCL_WORLD_COMM_GROUP, NCCL_WORLD_COMM_GROUP, \
    MCCL_WORLD_COMM_GROUP, _get_local_rank_helper, _get_local_size_helper, GlobalComm, \
    _check_mpi_envs, _set_elegant_exit_handle
from mindspore._c_expression import init_hccl, finalize_hccl, init_cluster, MSContext, ms_ctx_param


__all__ = ["init", "release", "get_rank", "get_local_rank", "get_group_size",
           "get_local_rank_size", "get_world_rank_from_group_rank",
           "get_group_rank_from_world_rank", "create_group", "destroy_group",
           "HCCL_WORLD_COMM_GROUP", "NCCL_WORLD_COMM_GROUP", "MCCL_WORLD_COMM_GROUP"]

DEFAULT_WORLD_COMM_GROUP = HCCL_WORLD_COMM_GROUP


def _set_rank_from_mpi():
    """Set environment variable according to OMPI"""
    ompi_rank_id = os.getenv("OMPI_COMM_WORLD_RANK")
    ompi_device_id = os.getenv("OMPI_COMM_WORLD_LOCAL_RANK")
    ompi_rank_size = os.getenv("OMPI_COMM_WORLD_SIZE")
    if ompi_rank_id:
        os.environ["RANK_ID"] = ompi_rank_id
    if ompi_device_id:
        os.environ["DEVICE_ID"] = ompi_device_id
        MSContext.get_instance().set_param(ms_ctx_param.device_id, int(ompi_device_id))
    if ompi_rank_size:
        os.environ["RANK_SIZE"] = ompi_rank_size


_set_rank_from_mpi()


def _get_group(group):
    """Return the world communication group if the `group` is `DEFAULT_WORLD_COMM_GROUP`."""
    if group == DEFAULT_WORLD_COMM_GROUP:
        return GlobalComm.WORLD_COMM_GROUP
    return group


def _host_distribute():
    """Check whether host distribute needed."""
    return os.getenv("MS_ROLE") or _check_mpi_envs()


def _check_parallel_envs():
    """
    Check whether parallel environment variables have been exported or not.

    Raises:
        RuntimeError: If parallel environment variables have not been exported or have been exported to wrong values.
    """
    if not GlobalComm.CHECK_ENVS:
        return
    rank_id_str = os.getenv("RANK_ID")
    if not rank_id_str:
        raise RuntimeError("Environment variables RANK_ID has not been exported, please export variables 'RANK_ID'.")
    try:
        int(rank_id_str)
    except ValueError:
        print("Environment variables 'RANK_ID' should be number, but got the type : {}".format(type(rank_id_str)))
    finally:
        pass
    rank_table_file_str = os.getenv("MINDSPORE_HCCL_CONFIG_PATH")
    rank_table_file_str_old = os.getenv("RANK_TABLE_FILE")
    help_cluster = os.getenv("HELP_CLUSTER")
    if not rank_table_file_str and not rank_table_file_str_old and not help_cluster:
        raise RuntimeError("Get hccl rank_table_file failed, "
                           "please export MINDSPORE_HCCL_CONFIG_PATH or RANK_TABLE_FILE.")


def init(backend_name=None):
    """
    Initialize distributed backends required by communication services, e.g. HCCL/NCCL. It is usually used in
    distributed parallel scenarios and set before using communication services.

    Note:
        - The full name of HCCL is Huawei Collective Communication Library.
        - The full name of NCCL is NVIDIA Collective Communication Library.
        - The full name of MCCL is MindSpore Collective Communication Library.

    Args:
        backend_name (str): Backend, using HCCL/NCCL/MCCL. HCCL should be used for Ascend hardware platforms and
                            NCCL for GPU hardware platforms. If not set, inference is automatically made based on the
                            hardware platform type (device_target). Default: None.

    Raises:
        TypeError: If `backend_name` is not a string.
        RuntimeError: If device target is invalid, or backend is invalid, or distributed initialization fails,
                      or the environment variables RANK_ID/MINDSPORE_HCCL_CONFIG_PATH
                      have not been exported when backend is HCCL.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `Ascend tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#preparations>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `GPU tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html#preparation>`_ .

        >>> from mindspore.communication import init
        >>> init()
    """
    host_init = _host_distribute()
    device_target = context.get_context("device_target")

    if backend_name is None:
        if device_target == "Ascend":
            backend_name = "hccl"
        elif device_target == "GPU":
            backend_name = "nccl"
        elif device_target == "CPU":
            backend_name = "mccl"
        else:
            raise RuntimeError("For 'set_context', the argument 'device_target' {} is not supported in "
                               "parallel initialization, please use Ascend, GPU or CPU.".format(device_target))
    if not isinstance(backend_name, str):
        raise TypeError("For 'init', the argument 'backend_name' must be a string, "
                        "but got the type : {}".format(type(backend_name)))

    if backend_name == "hccl":
        if _is_ps_mode():
            # Use MindSpore cluster to build network for Parameter Server traning.
            init_cluster()
            if _is_role_sched() or _is_role_pserver():
                raise RuntimeError("Parameter server and scheduler should use 'CPU' as backend instead of 'Ascend'")
            if _get_ps_context("worker_num") == 1:
                GlobalComm.INITED = True
                _set_elegant_exit_handle()
                return
        if device_target != "Ascend":
            raise RuntimeError("For 'init', the argument  'backend_name' should be 'Ascend' to init hccl, "
                               "but got {}".format(device_target))
        if not host_init:
            _check_parallel_envs()
        GlobalComm.BACKEND = Backend("hccl")
        init_hccl()
        GlobalComm.WORLD_COMM_GROUP = HCCL_WORLD_COMM_GROUP
    elif backend_name == "nccl":
        init_cluster()
        GlobalComm.WORLD_COMM_GROUP = NCCL_WORLD_COMM_GROUP
    elif backend_name == "mccl":
        init_cluster()
        GlobalComm.WORLD_COMM_GROUP = MCCL_WORLD_COMM_GROUP
    else:
        raise RuntimeError("For 'init', the argument 'backend_name' must be nccl while 'device_target' is GPU, "
                           "but got the 'backend_name' : hccl.")
    GlobalComm.INITED = True
    _set_elegant_exit_handle()


def release():
    """
    Release distributed resource. e.g. HCCL/NCCL.

    Note:
        This method should be used after init().

    Raises:
        RuntimeError: If failed to release distributed resource.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `Ascend tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#preparations>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `GPU tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html#preparation>`_ .

        >>> from mindspore.communication import init, release
        >>> init()
        >>> release()
    """
    finalize_hccl()


def get_rank(group=GlobalComm.WORLD_COMM_GROUP):
    """
    Get the rank ID for the current device in the specified collective communication group.

    Note:
        This method should be used after init().

    Args:
        group (str): The communication group to work on. Normally, the group should be created by create_group,
                     otherwise, using the default group. Default: WORLD_COMM_GROUP.

    Returns:
        int, the rank ID of the calling process within the group.

    Raises:
        TypeError: If group is not a string.
        ValueError: If backend is invalid.
        RuntimeError: If HCCL/NCCL is not available.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `Ascend tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#preparations>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `GPU tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html#preparation>`_ .

        >>> from mindspore.communication import init, get_rank
        >>> init()
        >>> rank_id = get_rank()
        >>> print(rank_id)
        >>> # the result is the rank_id in world_group
    """
    if not isinstance(group, str):
        raise TypeError("For 'get_rank', the argument 'group' must be type of string, "
                        "but got 'group' type : {}.".format(type(group)))
    return _get_rank_helper(group=_get_group(group))


def get_local_rank(group=GlobalComm.WORLD_COMM_GROUP):
    """
    Gets local rank ID for current device in specified collective communication group.

    Note:
        GPU version of MindSpore doesn't support this method.
        This method should be used after init().

    Args:
        group (str): The communication group to work on. Normally, the group should be created by create_group,
                     otherwise, using the default group. Default: WORLD_COMM_GROUP.

    Returns:
        int, the local rank ID of the calling process within the group.

    Raises:
        TypeError: If group is not a string.
        ValueError: If backend is invalid.
        RuntimeError: If HCCL is not available or MindSpore is GPU version.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `Ascend tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#preparations>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `GPU tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html#preparation>`_ .

        >>> import mindspore as ms
        >>> from mindspore.communication.management import init, get_rank, get_local_rank
        >>> ms.set_context(device_target="Ascend")
        >>> ms.set_auto_parallel_context(device_num=16) # 2 server, each server with 8 NPU.
        >>> init()
        >>> world_rank = get_rank()
        >>> local_rank = get_local_rank()
        >>> print("local_rank is: {}, world_rank is {}".format(local_rank, world_rank))
        local_rank is: 1, world_rank is 9
    """
    if not isinstance(group, str):
        raise TypeError("For 'get_local_rank', the argument 'group' must be type of string, "
                        "but got 'group' type : {}.".format(type(group)))
    return _get_local_rank_helper(group=_get_group(group))


def get_group_size(group=GlobalComm.WORLD_COMM_GROUP):
    """
    Get the rank size of the specified collective communication group.

    Note:
        This method should be used after init().

    Args:
        group (str): The communication group to work on. Normally, the group should be created by create_group,
                     otherwise, using the default group. Default: WORLD_COMM_GROUP.

    Returns:
        int, the rank size of the group.

    Raises:
        TypeError: If group is not a string.
        ValueError: If backend is invalid.
        RuntimeError: If HCCL/NCCL is not available.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `Ascend tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#preparations>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `GPU tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html#preparation>`_ .

        >>> import mindspore as ms
        >>> from mindspore.communication.management import init, get_group_size
        >>> ms.set_auto_parallel_context(device_num=8)
        >>> init()
        >>> group_size = get_group_size()
        >>> print("group_size is: ", group_size)
        group_size is: 8
    """
    if not isinstance(group, str):
        raise TypeError("For 'get_group_size', the argument 'group' must be type of string, "
                        "but got 'group' type : {}.".format(type(group)))
    return _get_size_helper(group=_get_group(group))


def get_local_rank_size(group=GlobalComm.WORLD_COMM_GROUP):
    """
    Gets local rank size of the specified collective communication group.

    Note:
        GPU version of MindSpore doesn't support this method.
        This method should be used after init().

    Args:
        group (str): The communication group to work on. The group is created by create_group
                     or the default world communication group. Default: WORLD_COMM_GROUP.

    Returns:
        int, the local rank size where the calling process is within the group.

    Raises:
        TypeError: If group is not a string.
        ValueError: If backend is invalid.
        RuntimeError: If HCCL is not available or MindSpore is GPU version.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `Ascend tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#preparations>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `GPU tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html#preparation>`_ .

        >>> import mindspore as ms
        >>> from mindspore.communication.management import init, get_local_rank_size
        >>> ms.set_context(device_target="Ascend")
        >>> ms.set_auto_parallel_context(device_num=16) # 2 server, each server with 8 NPU.
        >>> init()
        >>> local_rank_size = get_local_rank_size()
        >>> print("local_rank_size is: ", local_rank_size)
        local_rank_size is: 8
    """
    if not isinstance(group, str):
        raise TypeError("For 'get_local_rank_size', the argument 'group' must be type of string, "
                        "but got 'group' type : {}.".format(type(group)))
    return _get_local_size_helper(group=_get_group(group))


def get_world_rank_from_group_rank(group, group_rank_id):
    """
    Get the rank ID in the world communication group corresponding to
    the rank ID in the specified user communication group.

    Note:
        GPU version of MindSpore doesn't support this method.
        The parameter group should not be "hccl_world_group".
        This method should be used after init().

    Args:
        group (str): The communication group to work on. The group is created by create_group.
        group_rank_id (int): A rank ID in the communication group.

    Returns:
        int, the rank ID in world communication group.

    Raises:
        TypeError: If `group_rank_id` is not an integer or the group is not a string.
        ValueError: If group is 'hccl_world_group' or backend is invalid.
        RuntimeError: If HCCL is not available or MindSpore is GPU version.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `Ascend tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#preparations>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `GPU tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html#preparation>`_ .

        >>> from mindspore import set_context
        >>> from mindspore.communication.management import init, create_group, get_world_rank_from_group_rank
        >>> set_context(device_target="Ascend")
        >>> init()
        >>> group = "0-4"
        >>> rank_ids = [0,4]
        >>> create_group(group, rank_ids)
        >>> world_rank_id = get_world_rank_from_group_rank(group, 1)
        >>> print("world_rank_id is: ", world_rank_id)
        world_rank_id is: 4
    """
    if not isinstance(group, str):
        raise TypeError("For 'get_world_rank_from_group_rank', the argument 'group' must be type of string, "
                        "but got 'group' type : {}.".format(type(group)))
    return _get_world_rank_from_group_rank_helper(group=group, group_rank_id=group_rank_id)


def get_group_rank_from_world_rank(world_rank_id, group):
    """
    Get the rank ID in the specified user communication group corresponding to
    the rank ID in the world communication group.

    Note:
        GPU version of MindSpore doesn't support this method.
        The parameter group should not be "hccl_world_group".
        This method should be used after init().

    Args:
        world_rank_id (int): A rank ID in the world communication group.
        group (str): The communication group to work on. The group is created by create_group.

    Returns:
        int, the rank ID in the user communication group.

    Raises:
        TypeError: If world_rank_id is not an integer or the group is not a string.
        ValueError: If group is 'hccl_world_group' or backend is invalid.
        RuntimeError: If HCCL is not available or MindSpore is GPU version.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `Ascend tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#preparations>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `GPU tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html#preparation>`_ .

        >>> from mindspore import set_context
        >>> from mindspore.communication.management import init, create_group, get_group_rank_from_world_rank
        >>> set_context(device_target="Ascend")
        >>> init()
        >>> group = "0-4"
        >>> rank_ids = [0,4]
        >>> create_group(group, rank_ids)
        >>> group_rank_id = get_group_rank_from_world_rank(4, group)
        >>> print("group_rank_id is: ", group_rank_id)
        group_rank_id is: 1
    """
    if not isinstance(group, str):
        raise TypeError("For 'get_group_rank_from_world_rank', the argument 'group' must be type of string, "
                        "but got 'group' type : {}.".format(type(group)))
    return _get_group_rank_from_world_rank_helper(world_rank_id=world_rank_id, group=group)


def create_group(group, rank_ids):
    """
    Create a user collective communication group.

    Note:
        GPU version of MindSpore doesn't support this method.
        The size of rank_ids should be larger than 1, rank_ids should not have duplicate data.
        This method should be used after init().
        Only support global single communication group in PyNative mode if you do not start with mpirun.

    Args:
        group (str): The name of the communication group to be created.
        rank_ids (list): A list of device IDs.

    Raises:
        TypeError: If group is not a string or `rank_ids` is not a list.
        ValueError: If `rank_ids` size is not larger than 1, or `rank_ids` has duplicate data, or backend is invalid.
        RuntimeError: If HCCL is not available or MindSpore is GPU version.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `Ascend tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#preparations>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `GPU tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html#preparation>`_ .

        >>> from mindspore import set_context
        >>> import mindspore.ops as ops
        >>> from mindspore.communication.management import init, create_group
        >>> set_context(device_target="Ascend")
        >>> init()
        >>> group = "0-8"
        >>> rank_ids = [0,8]
        >>> create_group(group, rank_ids)
        >>> allreduce = ops.AllReduce(group)
    """
    if not isinstance(group, str):
        raise TypeError("For 'create_group', the argument 'group' must be type of string, "
                        "but got 'group' type : {}.".format(type(group)))
    _create_group_helper(group, rank_ids)


def destroy_group(group):
    """
    Destroy the user collective communication group.

    Note:
        GPU version of MindSpore doesn't support this method.
        The parameter group should not be "hccl_world_group".
        This method should be used after init().

    Args:
        group (str): The communication group to destroy, the group should be created by create_group.

    Raises:
        TypeError: If group is not a string.
        ValueError: If group is "hccl_world_group" or backend is invalid.
        RuntimeError: If HCCL is not available or MindSpore is GPU version.

    Supported Platforms:
        ``Ascend``
    """
    if not isinstance(group, str):
        raise TypeError("For 'destroy_group', the argument 'group' must be type of string, "
                        "but got 'group' type : {}.".format(type(group)))
    _destroy_group_helper(group)
