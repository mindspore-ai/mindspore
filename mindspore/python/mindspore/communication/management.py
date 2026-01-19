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
from mindspore import log as logger
from mindspore.communication._comm_helper import Backend, _get_rank_helper, _get_size_helper, \
    _get_world_rank_from_group_rank_helper, _get_group_rank_from_world_rank_helper, \
    _create_group_helper, _destroy_group_helper, HCCL_WORLD_COMM_GROUP, NCCL_WORLD_COMM_GROUP, \
    MCCL_WORLD_COMM_GROUP, DEVICE_TO_BACKEND, _get_local_rank_helper, _get_local_size_helper, GlobalComm, \
    _check_mpi_envs, _set_elegant_exit_handle, _get_group_ranks, _get_comm_name_helper, _comm_switch_nic_helper
from mindspore._c_expression import init_hccl, finalize_hccl, init_cluster, MSContext, ms_ctx_param, \
    _init_hccl_with_store, _init_cluster_with_store
from mindspore.hal.device import is_initialized

__all__ = ["init", "release", "get_rank", "get_local_rank", "get_group_size",
           "get_local_rank_size", "get_world_rank_from_group_rank", "get_comm_name",
           "get_group_rank_from_world_rank", "create_group", "destroy_group", "get_process_group_ranks",
           "HCCL_WORLD_COMM_GROUP", "NCCL_WORLD_COMM_GROUP", "MCCL_WORLD_COMM_GROUP"]

DEFAULT_WORLD_COMM_GROUP = HCCL_WORLD_COMM_GROUP


def _set_rank_from_mpi():
    """Set environment variable according to OMPI"""
    ompi_rank_id = os.getenv("OMPI_COMM_WORLD_RANK")
    ompi_device_id = os.getenv("OMPI_COMM_WORLD_LOCAL_RANK")
    ompi_rank_size = os.getenv("OMPI_COMM_WORLD_SIZE")
    if ompi_rank_id and os.getenv("MS_ROLE"):
        logger.warning("Launching distributed job using both dynamic cluster and OpenMPI at the same time. "
                       "MindSpore will prioritize the use of dynamic cluster. Do not set env from OpenMPI.")
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
    compile_level = os.getenv("MS_SIMULATION_LEVEL")
    if compile_level:
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


def _set_envs():
    """
    Some environmental variables must be set after `init` is completed. This takes compatibility
    into account because user scripts may get 'DEVICE_ID' or 'RANK_ID' envs using rank table.
    """
    if not os.getenv("RANK_TABLE_FILE"):
        return

    os.environ["RANK_ID"] = str(get_rank())
    os.environ["DEVICE_ID"] = str(context.get_context("device_id"))
    if os.getenv("RANK_SIZE") is None:
        os.environ["RANK_SIZE"] = str(get_group_size())


def _check_hccl():
    """Check hcll is installed needed."""
    if not GlobalComm.CHECK_ENVS:
        return
    try:
        from hccl import sys_version as hccl_version
        v = '.'.join(hccl_version.__sys_version__.split('.')[0:2])
        logger.debug(f"\"hccl\" wheel package version {v} is installed")
    except Exception as e:
        logger.error(f"Check hccl failed: {e}")
        raise RuntimeError("\"hccl\" wheel was not installed correctly. For details, refer to the installation "
                           "guidelines: https://www.mindspore.cn/install") from e


def init(backend_name=None):
    """
    Initialize distributed backends required by communication services, e.g. ``"hccl"`` / ``"nccl"`` / ``"mccl"``.
    It is usually used in distributed parallel scenarios and set before using communication services.

    Note:
        - The full name of ``"hccl"`` is Huawei Collective Communication Library(HCCL).
        - The full name of ``"nccl"`` is NVIDIA Collective Communication Library(NCCL).
        - The full name of ``"mccl"`` is MindSpore Collective Communication Library(MCCL).
        - In Ascend hardware platforms, ``init()`` should be set before the definition of any Tensor and Parameter,
          and the instantiation and execution of any operation and net.

    Args:
        backend_name (str, optional): Backend, using ``"hccl"`` / ``"nccl"`` / ``"mccl"``.
            ``"hccl"`` should be used for Ascend hardware platforms,
            ``"nccl"`` for GPU hardware platforms and ``"mccl"`` for CPU hardware platforms.
            If not set, inference is automatically made based on the hardware
            platform type (device_target). Default: ``None`` .

    Raises:
        TypeError: If `backend_name` is not a string.
        RuntimeError: If device target is invalid, or backend is invalid, or distributed initialization fails.
        RuntimeError: If the environment variables RANK_ID/MINDSPORE_HCCL_CONFIG_PATH
                      have not been exported when backend is HCCL.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun startup
            <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_
            for more details.

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
            if os.getenv("MS_SIMULATION_LEVEL"):
                raise RuntimeError("Dryrun is not supported on CPU device for a distributed job.")
        else:
            raise RuntimeError("For 'set_context', the argument 'device_target' {} is not supported in "
                               "parallel initialization, please use Ascend, GPU or CPU.".format(device_target))
    if not isinstance(backend_name, str):
        raise TypeError("For 'init', the argument 'backend_name' must be a string, "
                        "but got the type : {}".format(type(backend_name)))
    if os.getenv("MS_ROLE") == "MS_SCHED":
        backend_name = "mccl"

    _set_elegant_exit_handle()
    if backend_name == "hccl":
        if device_target != "Ascend":
            raise RuntimeError("For 'init', the argument 'backend_name' should be '{}' to init '{}', "
                               "but got 'hccl'.".format(DEVICE_TO_BACKEND[device_target], device_target))
        if is_initialized(device_target):
            logger.warning("For 'init' in Ascend backend, the backend is already initialized, please set it before "
                           "the definition of any Tensor and Parameter, and the instantiation and execution of any "
                           "operation and net, otherwise the 'init' may not take effect.")
        if not host_init:
            _check_parallel_envs()
        GlobalComm.BACKEND = Backend("hccl")
        _check_hccl()
        init_hccl()
        GlobalComm.WORLD_COMM_GROUP = HCCL_WORLD_COMM_GROUP
    elif backend_name == "nccl":
        if device_target != "GPU":
            raise RuntimeError("For 'init', the argument 'backend_name' should be '{}' to init '{}', "
                               "but got 'nccl'.".format(DEVICE_TO_BACKEND[device_target], device_target))
        init_cluster()
        GlobalComm.BACKEND = Backend("nccl")
        GlobalComm.WORLD_COMM_GROUP = NCCL_WORLD_COMM_GROUP
    elif backend_name == "mccl":
        init_cluster()
        GlobalComm.BACKEND = Backend("mccl")
        GlobalComm.WORLD_COMM_GROUP = MCCL_WORLD_COMM_GROUP
    else:
        raise RuntimeError("For 'init', the argument 'backend_name' must be one of 'hccl', 'nccl' and 'mccl', "
                           "but got 'backend_name' : {}".format(backend_name))

    GlobalComm.INITED = True
    _set_envs()


def _init_without_sched(backend_name=None, init_method=None, timeout=None, world_size=-1, rank=-1, store=None):
    """
    Initialize the distributed backends required by the communication services through an existing TcpStore or
    by creating a new TcpStore. This approach does not rely on an additional Scheduler process.

    Args:
        backend_name (str, optional): Backend, using ``"hccl"`` / ``"nccl"`` / ``"mccl"``.
            ``"hccl"`` should be used for Ascend hardware platforms,
            ``"nccl"`` for GPU hardware platforms and ``"mccl"`` for CPU hardware platforms.
            If not set, inference is automatically made based on the hardware platform type (device_target).
            Default: ``None`` .
        init_method (str, optional): URL specifying how to init collective communication group. Default is ``None``.
        timeout (timedelta, optional): Timeout for API executed. Default is ``None``. Currently, this parameter is
            only supported for host-side cluster network configuration using `init_method` or `store`.
        world_size (int, optional): Number of the processes participating in the job. Default is ``-1``.
        rank (int, optional): Rank of the current process. Default is ``-1``.
        store (Store, optional): An object that stores key/value data, facilitating the exchange of inter-process
            communication addresses and connection information. Default is ``None``. Currently, only the
            ``TCPStore`` type is supported.

    Raises:
        TypeError: If `backend_name` is not a string.
        RuntimeError: If device target is invalid, or backend is invalid, or distributed initialization fails,
                      or the environment variables RANK_ID/MINDSPORE_HCCL_CONFIG_PATH
                      have not been exported when backend is HCCL.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
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

    _set_elegant_exit_handle()
    if backend_name == "hccl":
        if device_target != "Ascend":
            raise RuntimeError("For 'init', the argument 'backend_name' should be '{}' to init '{}', "
                               "but got 'hccl'.".format(DEVICE_TO_BACKEND[device_target], device_target))
        if is_initialized(device_target):
            logger.warning("For 'init' in Ascend backend, the backend is already initialized, please set it before "
                           "the definition of any Tensor and Parameter, and the instantiation and execution of any "
                           "operation and net, otherwise the 'init' may not take effect.")
        GlobalComm.BACKEND = Backend("hccl")
        _check_hccl()
        _init_hccl_with_store(init_method, timeout, world_size, rank, store)
        GlobalComm.WORLD_COMM_GROUP = HCCL_WORLD_COMM_GROUP
    elif backend_name == "nccl":
        if device_target != "GPU":
            raise RuntimeError("For 'init', the argument 'backend_name' should be '{}' to init '{}', "
                               "but got 'nccl'.".format(DEVICE_TO_BACKEND[device_target], device_target))
        _init_cluster_with_store(init_method, timeout, world_size, rank, store)
        GlobalComm.BACKEND = Backend("nccl")
        GlobalComm.WORLD_COMM_GROUP = NCCL_WORLD_COMM_GROUP
    elif backend_name == "mccl":
        _init_cluster_with_store(init_method, timeout, world_size, rank, store)
        GlobalComm.BACKEND = Backend("mccl")
        GlobalComm.WORLD_COMM_GROUP = MCCL_WORLD_COMM_GROUP
    else:
        raise RuntimeError("For 'init', the argument 'backend_name' must be one of 'hccl', 'nccl' and 'mccl', "
                           "but got 'backend_name' : {}".format(backend_name))

    GlobalComm.INITED = True
    _set_envs()


def release():
    """
    Release distributed resource. e.g. HCCL/NCCL/MCCL.

    Note:
        This method should be used after `init()` . If not, resource will be released when program ends.

    Raises:
        RuntimeError: If failed to release distributed resource.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun startup
            <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_
            for more details.

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
        group (str, optional): The communication group to work on. Normally, the group should be created by
            create_group, otherwise, using the default group. Default: ``GlobalComm.WORLD_COMM_GROUP`` .

    Returns:
        int, the rank ID of the calling process within the group.

    Raises:
        TypeError: If group is not a string.
        ValueError: If backend is invalid.
        RuntimeError: If HCCL/NCCL/MCCL is not available.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun startup
            <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_
            for more details.

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
        This method isn't supported in GPU and CPU versions of MindSpore.
        This method should be used after init().

    Args:
        group (str, optional): The communication group to work on. Normally, the group should be created by
            create_group, otherwise, using the default group. Default: ``GlobalComm.WORLD_COMM_GROUP``.

    Returns:
        int, the local rank ID of the calling process within the group.

    Raises:
        TypeError: If group is not a string.
        ValueError: If backend is invalid.
        RuntimeError: If HCCL is not available or MindSpore is GPU/CPU version.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun startup
            <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_
            for more details.

        >>> import mindspore as ms
        >>> from mindspore.communication import init, get_rank, get_local_rank
        >>> ms.set_device(device_target="Ascend")
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
        group (str, optional): The communication group to work on. Normally, the group should be created by
            create_group, otherwise, using the default group. Default: ``GlobalComm.WORLD_COMM_GROUP``.

    Returns:
        int, the rank size of the group.

    Raises:
        TypeError: If group is not a string.
        ValueError: If backend is invalid.
        RuntimeError: If HCCL/NCCL/MCCL is not available.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun startup
            <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_
            for more details.

        >>> import mindspore as ms
        >>> from mindspore.communication import init, get_group_size
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
        This method isn't supported in GPU and CPU versions of MindSpore.
        This method should be used after init().

    Args:
        group (str, optional): The communication group to work on. The group is created by create_group
            or the default world communication group. Default: ``GlobalComm.WORLD_COMM_GROUP`` .

    Returns:
        int, the local rank size where the calling process is within the group.

    Raises:
        TypeError: If group is not a string.
        ValueError: If backend is invalid.
        RuntimeError: If HCCL is not available or MindSpore is GPU/CPU version.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun startup
            <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_
            for more details.

        >>> import mindspore as ms
        >>> from mindspore.communication import init, get_local_rank_size
        >>> ms.set_device(device_target="Ascend")
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
        This method isn't supported in GPU and CPU versions of MindSpore.
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
        RuntimeError: If HCCL is not available or MindSpore is GPU/CPU version.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun startup
            <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_
            for more details.

        >>> import mindspore as ms
        >>> from mindspore import set_context
        >>> from mindspore.communication import init, create_group, get_world_rank_from_group_rank, get_rank
        >>> set_context(mode=ms.GRAPH_MODE)
        >>> ms.set_device(device_target="Ascend")
        >>> init()
        >>> group = "0-4"
        >>> rank_ids = [0,4]
        >>> if get_rank() in rank_ids:
        ...     create_group(group, rank_ids)
        ...     world_rank_id = get_world_rank_from_group_rank(group, 1)
        ...     print("world_rank_id is: ", world_rank_id)
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
        This method isn't supported in GPU and CPU versions of MindSpore.
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
        RuntimeError: If HCCL is not available or MindSpore is GPU/CPU version.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun startup
            <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_
            for more details.

        >>> import mindspore as ms
        >>> from mindspore import set_context
        >>> from mindspore.communication import init, create_group, get_group_rank_from_world_rank, get_rank
        >>> set_context(mode=ms.GRAPH_MODE)
        >>> ms.set_device(device_target="Ascend")
        >>> init()
        >>> group = "0-4"
        >>> rank_ids = [0,4]
        >>> if get_rank() in rank_ids:
        ...     create_group(group, rank_ids)
        ...     group_rank_id = get_group_rank_from_world_rank(4, group)
        ...     print("group_rank_id is: ", group_rank_id)
        group_rank_id is: 1
    """
    if not isinstance(group, str):
        raise TypeError("For 'get_group_rank_from_world_rank', the argument 'group' must be type of string, "
                        "but got 'group' type : {}.".format(type(group)))
    return _get_group_rank_from_world_rank_helper(world_rank_id=world_rank_id, group=group)


def create_group(group, rank_ids, options=None):
    """
    Create a user collective communication group.

    Note:
        This method isn't supported in GPU and CPU versions of MindSpore.
        The size of rank_ids should be larger than 1, rank_ids should not have duplicate data.
        This method should be used after init().
        Only support global single communication group in PyNative mode if you do not start with mpirun.

    Args:
        group (str): The name of the communication group to be created.
        rank_ids (list): A list of device IDs.
        options (GroupOptions, optional): Additional communication group configuration parameters.
            The backend will automatically select supported parameters and apply them during group
            initialization. i.e. for the ``HCCL`` backend, ``hccl_config`` can be specified so that
            group initialization configurations can be applied. Default is ``None``.

            `GroupOptions` is defined as a class that can be instantiated as a python object.

            .. code-block::

                GroupOptions {
                    hccl_config(dict)
                }

            `hccl_config` currently only supports "hccl_buffer_size" or "hccl_comm".

            - hccl_buffer_size (uint32): specifies the size of the HCCL communication buffer.
            - hccl_comm (int64): specifies an existing HcclComm pointer. If "hccl_comm" is set,
              "hccl_buffer_size" will be ignored.

    Raises:
        TypeError: If group is not a string or `rank_ids` is not a list.
        ValueError: If `rank_ids` size is not larger than 1, or `rank_ids` has duplicate data, or backend is invalid.
        RuntimeError: If HCCL is not available or MindSpore is GPU/CPU version.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun startup
            <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_
            for more details.

        >>> import mindspore as ms
        >>> from mindspore import set_context, ops
        >>> from mindspore._c_expression import GroupOptions
        >>> from mindspore.communication import init, create_group, get_rank
        >>> set_context(mode=ms.GRAPH_MODE)
        >>> ms.set_device(device_target="Ascend")
        >>> init()
        >>> group = "0-7"
        >>> rank_ids = [0,7]
        >>> options = GroupOptions()
        >>> options.hccl_config = {"hccl_buffer_size": 400}
        >>> if get_rank() in rank_ids:
        ...     create_group(group, rank_ids, options)
        ...     allreduce = ops.AllReduce(group)
    """
    if not isinstance(group, str):
        raise TypeError("For 'create_group', the argument 'group' must be type of string, "
                        "but got 'group' type : {}.".format(type(group)))
    _create_group_helper(group, rank_ids, options)


def destroy_group(group):
    """
    Destroy the user collective communication group.

    Note:
        This method isn't supported in GPU and CPU versions of MindSpore.
        The parameter group should not be "hccl_world_group".
        This method should be used after init().

    Args:
        group (str): The communication group to destroy, the group should be created by create_group.

    Raises:
        TypeError: If group is not a string.
        ValueError: If group is "hccl_world_group" or backend is invalid.
        RuntimeError: If HCCL is not available or MindSpore is GPU/CPU version.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun startup
            <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_
            for more details.

        >>> import mindspore as ms
        >>> from mindspore import set_context
        >>> from mindspore import ops
        >>> from mindspore.communication import init, create_group, destroy_group, get_rank
        >>> set_context(mode=ms.GRAPH_MODE)
        >>> ms.set_device(device_target="Ascend")
        >>> init()
        >>> group = "0-2"
        >>> rank_ids = [0,2]
        >>> if get_rank() in rank_ids:
        ...     create_group(group, rank_ids)
        ...     destroy_group(group)
    """
    if not isinstance(group, str):
        raise TypeError("For 'destroy_group', the argument 'group' must be type of string, "
                        "but got 'group' type : {}.".format(type(group)))
    _destroy_group_helper(group)


def get_comm_name(group=GlobalComm.WORLD_COMM_GROUP):
    """
    Get the communicator name of the specified collective communication group.

    Note:
        This method isn't supported in GPU and CPU versions of MindSpore.
        This method should be used after init().

    Args:
        group (str, optional): The communication group to work on. Normally, the group should be created by
            create_group, otherwise, using the default group. Default: ``GlobalComm.WORLD_COMM_GROUP`` .

    Returns:
        string, the inner communicator name of the group.

    Raises:
        TypeError: If group is not a string.
        ValueError: If backend is invalid.
        RuntimeError: If HCCL is not available or MindSpore is GPU/CPU version.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun startup
            <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_
            for more details.

        >>> import mindspore as ms
        >>> from mindspore.communication import init, create_group, get_rank, get_comm_name
        >>> ms.set_device(device_target="Ascend")
        >>> init()
        >>> world_group_comm_name = get_comm_name()
        >>> group = "0-7"
        >>> rank_ids = [0,7]
        >>> if get_rank() in rank_ids:
        ...     create_group(group, rank_ids)
        ...     customizd_group_comm_name = get_comm_name(group)
        ...     print("comm_name of customizd group is ", customizd_group_comm_name)
        >>> print("comm_name of world group is: ", world_group_comm_name)
        comm_name of customizd group is: 11.22.33.44%eth0_60000_0_0123456789101112
        comm_name of world group is: 11.22.33.44%eth0_60000_0_1211109876543210
    """
    if not isinstance(group, str):
        raise TypeError("For 'get_comm_name', the argument 'group' must be type of string, "
                        "but got 'group' type : {}.".format(type(group)))
    return _get_comm_name_helper(group=_get_group(group))


def get_process_group_ranks(group=GlobalComm.WORLD_COMM_GROUP):
    """
    Gets the ranks of the specific group and returns the process ranks in the communication group as a list.

    Args:
        group (str, optional): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP`` , which
            means ``"hccl_world_group"`` in Ascend, and ``"nccl_world_group"`` in GPU.

    Returns:
        List (List[int]), List of process ranks in the specified communication group.

    Raises:
        TypeError: If the `group` is not a str.
        RuntimeError: If device target is invalid, or backend is invalid, or distributed initialization fails.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.

            Please see the `msrun startup
            <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_
            for more details.

            This example should be run with 4 devices.

        >>> import numpy as np
        >>> from mindspore.communication import init, get_process_group_ranks
        >>>
        >>> init()
        >>> output = get_process_group_ranks()
        >>> print(output)
        [0, 1, 2, 3]

    """
    return _get_group_ranks(group=_get_group(group))


def _comm_switch_nic(global_ranks, use_backup):
    """Switch network interface card between the primary and the secondary NIC.

    Args:
        global_ranks (list[int], tuple[int]): list of integers. The global rank ids that need switch network interface .
        use_backup (list[bool], tuple[int]): list of bool. For each rank id in global_ranks, determine whether to use
            the backup network interface card. True means use, False means not use.

    Returns:
        bool, whether the network card switch is successful.
            If one fails, return False. If all are successful, return True.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.

            Please see the `msrun startup
            <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_
            for more details.

            This example should be run with 4 devices.

        >>> import numpy as np
        >>> from mindspore.communication import init, _comm_switch_nic
        >>> from mindspore.communication.management import _comm_switch_nic
        >>>
        >>> init()
        >>> ret = _comm_switch_nic([0, 1], [True, False])
        >>> print(ret)
        True

    """
    max_rank = get_group_size() - 1
    if not all(isinstance(i, (list, tuple)) for i in (global_ranks, use_backup)):
        raise ValueError(f"For _comm_switch_nic, the args 'global_ranks' and 'use_backup' should be list or tuple, "
                         f"but got 'global_ranks' type {type(global_ranks)}, 'use_backup' type {type(use_backup)}")
    if not all(isinstance(rank, int) and not isinstance(rank, bool) and rank <= max_rank for rank in global_ranks):
        raise ValueError(f"For _comm_switch_nic, the all elements  in 'global_ranks' should be int number, and less "
                         f"than {get_group_size()}, but got 'global_ranks' : {global_ranks}.")
    if not all(isinstance(ub, bool) for ub in use_backup):
        raise ValueError(f"For _comm_switch_nic, the all elements  in 'use_backup' should be bool, but got "
                         f"'use_backup' : {use_backup}.")
    if len(set(global_ranks)) != len(global_ranks):
        raise ValueError(f"For _comm_switch_nic, the all elements  in 'global_ranks' should be different, but got "
                         f"'global_ranks' : {global_ranks}.")
    if len(global_ranks) != len(use_backup):
        raise ValueError(f"For _comm_switch_nic, the elements number in 'global_ranks' should be equal to 'use_backup',"
                         f" but got 'global_ranks' {len(global_ranks)} elements: {global_ranks},"
                         f" 'use_backup' {len(use_backup)} elements:  {use_backup},.")
    return _comm_switch_nic_helper(global_ranks, use_backup)
