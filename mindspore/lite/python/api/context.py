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
"""
Context API.
"""
from ._checkparam import check_isinstance, check_list_of_element
from .lib import _c_lite_wrapper

__all__ = ['Context', 'DeviceInfo', 'CPUDeviceInfo', 'GPUDeviceInfo', 'AscendDeviceInfo']


class Context:
    """
    Context is used to store environment variables during execution.

    The context should be configured before running the program.
    If it is not configured, it will be automatically set according to the device target by default.

    Note:
        If core_list and mode are set by SetThreadAffinity at the same time, the core_list is effective, but the mode
        is not effective.
        If the default value of the parameter is none, it means the parameter is not set.

    Args:
        thread_num (int, optional): Set the number of threads at runtime. Default: None.
        inter_op_parallel_num (int, optional): Set the parallel number of operators at runtime. Default: None.
        thread_affinity_mode (int, optional): Set the thread affinity to CPU cores. Default: None.

            - 0: no affinities.
            - 1: big cores first.
            - 2: little cores first.

        thread_affinity_core_list (list[int], optional): Set the thread lists to CPU cores. Default: None.
        enable_parallel (bool, optional): Set the status whether to perform model inference or training in parallel.
            Default: False.

    Raises:
        TypeError: `thread_num` is neither an int nor None.
        TypeError: `inter_op_parallel_num` is neither an int nor None.
        TypeError: `thread_affinity_mode` is neither an int nor None.
        TypeError: `thread_affinity_core_list` is neither a list nor None.
        TypeError: `thread_affinity_core_list` is a list, but the elements are neither int nor None.
        TypeError: `enable_parallel` is not a bool.
        ValueError: `thread_num` is less than 0.
        ValueError: `inter_op_parallel_num` is less than 0.

    Examples:
        >>> import mindspore_lite as mslite
        >>> context = mslite.Context(thread_num=1, inter_op_parallel_num=1, thread_affinity_mode=1,
        ...                          enable_parallel=False)
        >>> print(context)
        thread_num: 1,
        inter_op_parallel_num: 1,
        thread_affinity_mode: 1,
        thread_affinity_core_list: [],
        enable_parallel: False,
        device_list: .
    """

    def __init__(self, thread_num=None, inter_op_parallel_num=None, thread_affinity_mode=None, \
                 thread_affinity_core_list=None, enable_parallel=False):
        if thread_num is not None:
            check_isinstance("thread_num", thread_num, int)
            if thread_num < 0:
                raise ValueError(f"Context's init failed, thread_num must be positive.")
        if inter_op_parallel_num is not None:
            check_isinstance("inter_op_parallel_num", inter_op_parallel_num, int)
            if inter_op_parallel_num < 0:
                raise ValueError(f"Context's init failed, inter_op_parallel_num must be positive.")
        if thread_affinity_mode is not None:
            check_isinstance("thread_affinity_mode", thread_affinity_mode, int)
        check_list_of_element("thread_affinity_core_list", thread_affinity_core_list, int, enable_none=True)
        check_isinstance("enable_parallel", enable_parallel, bool)
        core_list = [] if thread_affinity_core_list is None else thread_affinity_core_list
        self._context = _c_lite_wrapper.ContextBind()
        if thread_num is not None:
            self._context.set_thread_num(thread_num)
        if inter_op_parallel_num is not None:
            self._context.set_inter_op_parallel_num(inter_op_parallel_num)
        if thread_affinity_mode is not None:
            self._context.set_thread_affinity_mode(thread_affinity_mode)
        self._context.set_thread_affinity_core_list(core_list)
        self._context.set_enable_parallel(enable_parallel)

    def __str__(self):
        res = f"thread_num: {self._context.get_thread_num()},\n" \
              f"inter_op_parallel_num: {self._context.get_inter_op_parallel_num()},\n" \
              f"thread_affinity_mode: {self._context.get_thread_affinity_mode()},\n" \
              f"thread_affinity_core_list: {self._context.get_thread_affinity_core_list()},\n" \
              f"enable_parallel: {self._context.get_enable_parallel()},\n" \
              f"device_list: {self._context.get_device_list()}."
        return res

    def append_device_info(self, device_info):
        """
        Append one user-defined device info to the context.

        Note:
            After gpu device info is added, cpu device info must be added before call context.
            Because when ops are not supported on GPU, The system will try whether the CPU supports it.
            At that time, need to switch to the context with cpu device info.

            After Ascend device info is added, cpu device info must be added before call context.
            Because when ops are not supported on Ascend, The system will try whether the CPU supports it.
            At that time, need to switch to the context with cpu device info.

        Args:
            device_info (DeviceInfo): the instance of device info.

        Raises:
            TypeError: `device_info` is not a DeviceInfo.

        Examples:
            >>> import mindspore_lite as mslite
            >>> context = mslite.Context()
            >>> context.append_device_info(mslite.CPUDeviceInfo())
            >>> print(context)
            thread_num: 0,
            inter_op_parallel_num: 0,
            thread_affinity_mode: 0,
            thread_affinity_core_list: [],
            enable_parallel: False,
            device_list: 0, .
        """
        if not isinstance(device_info, DeviceInfo):
            raise TypeError("device_info must be DeviceInfo, but got {}.".format(
                type(device_info)))
        self._context.append_device_info(device_info._device_info)


class DeviceInfo:
    """
    DeviceInfo base class.
    """

    def __init__(self):
        """ Initialize DeviceInfo"""


class CPUDeviceInfo(DeviceInfo):
    """
    Helper class to set cpu device info, and it inherits DeviceInfo base class.

    Args:
        enable_fp16(bool, optional): enables to perform the float16 inference. Default: False.

    Raises:
        TypeError: `enable_fp16` is not a bool.

    Examples:
        >>> import mindspore_lite as mslite
        >>> cpu_device_info = mslite.CPUDeviceInfo(enable_fp16=True)
        >>> print(cpu_device_info)
        device_type: DeviceType.kCPU,
        enable_fp16: True.
        >>> context = mslite.Context()
        >>> context.append_device_info(cpu_device_info)
        >>> print(context)
        thread_num: 0,
        inter_op_parallel_num: 0,
        thread_affinity_mode: 0,
        thread_affinity_core_list: [],
        enable_parallel: False,
        device_list: 0, .
    """

    def __init__(self, enable_fp16=False):
        super(CPUDeviceInfo, self).__init__()
        check_isinstance("enable_fp16", enable_fp16, bool)
        self._device_info = _c_lite_wrapper.CPUDeviceInfoBind()
        self._device_info.set_enable_fp16(enable_fp16)

    def __str__(self):
        res = f"device_type: {self._device_info.get_device_type()},\n" \
              f"enable_fp16: {self._device_info.get_enable_fp16()}."
        return res


class GPUDeviceInfo(DeviceInfo):
    """
    Helper class to set gpu device info, and it inherits DeviceInfo base class.

    Args:
        device_id(int, optional): The device id. Default: 0.
        enable_fp16(bool, optional): enables to perform the float16 inference. Default: False.

    Raises:
        TypeError: `device_id` is not an int.
        TypeError: `enable_fp16` is not a bool.
        ValueError: `device_id` is less than 0.

    Examples:
        >>> import mindspore_lite as mslite
        >>> gpu_device_info = mslite.GPUDeviceInfo(device_id=1, enable_fp16=False)
        >>> print(gpu_device_info)
        device_type: DeviceType.kGPU,
        device_id: 1,
        enable_fp16: False.
        >>> cpu_device_info = mslite.CPUDeviceInfo(enable_fp16=False)
        >>> context = mslite.Context()
        >>> context.append_device_info(gpu_device_info)
        >>> context.append_device_info(cpu_device_info)
        >>> print(context)
        thread_num: 0,
        inter_op_parallel_num: 0,
        thread_affinity_mode: 0,
        thread_affinity_core_list: [],
        enable_parallel: False,
        device_list: 1, 0, .
    """

    def __init__(self, device_id=0, enable_fp16=False):
        super(GPUDeviceInfo, self).__init__()
        check_isinstance("device_id", device_id, int)
        if device_id < 0:
            raise ValueError(f"GPUDeviceInfo's init failed, device_id must be positive.")
        check_isinstance("enable_fp16", enable_fp16, bool)
        self._device_info = _c_lite_wrapper.GPUDeviceInfoBind()
        self._device_info.set_device_id(device_id)
        self._device_info.set_enable_fp16(enable_fp16)

    def __str__(self):
        res = f"device_type: {self._device_info.get_device_type()},\n" \
              f"device_id: {self._device_info.get_device_id()},\n" \
              f"enable_fp16: {self._device_info.get_enable_fp16()}."
        return res

    def get_rank_id(self):
        """
        Get the ID of the current device in the cluster from context.

        Returns:
            int, the ID of the current device in the cluster, which starts from 0.

        Examples:
            >>> import mindspore_lite as mslite
            >>> device_info = mslite.GPUDeviceInfo(device_id=1, enable_fp16=True)
            >>> rank_id = device_info.get_rank_id()
            >>> print(rank_id)
            0
        """
        return self._device_info.get_rank_id()

    def get_group_size(self):
        """
        Get the number of the clusters from context.

        Returns:
            int, the number of the clusters.

        Examples:
            >>> import mindspore_lite as mslite
            >>> device_info = mslite.GPUDeviceInfo(device_id=1, enable_fp16=True)
            >>> group_size = device_info.get_group_size()
            >>> print(group_size)
            1
        """
        return self._device_info.get_group_size()


class AscendDeviceInfo(DeviceInfo):
    """
    Helper class to set Ascend device infos, and it inherits DeviceInfo base class.

    Args:
        device_id(int, optional): The device id. Default: 0.

    Raises:
        TypeError: `device_id` is not an int.
        ValueError: `device_id` is less than 0.

    Examples:
        >>> import mindspore_lite as mslite
        >>> ascend_device_info = mslite.AscendDeviceInfo(device_id=0)
        >>> print(ascend_device_info)
        device_type: DeviceType.kAscend,
        device_id: 0.
        >>> cpu_device_info = mslite.CPUDeviceInfo(enable_fp16=False)
        >>> context = mslite.Context()
        >>> context.append_device_info(ascend_device_info)
        >>> context.append_device_info(cpu_device_info)
        >>> print(context)
        thread_num: 0,
        inter_op_parallel_num: 0,
        thread_affinity_mode: 0,
        thread_affinity_core_list: [],
        enable_parallel: False,
        device_list: 3, 0, .
    """

    def __init__(self, device_id=0):
        super(AscendDeviceInfo, self).__init__()
        check_isinstance("device_id", device_id, int)
        if device_id < 0:
            raise ValueError(f"AscendDeviceInfo's init failed, device_id must be positive.")
        self._device_info = _c_lite_wrapper.AscendDeviceInfoBind()
        self._device_info.set_device_id(device_id)

    def __str__(self):
        res = f"device_type: {self._device_info.get_device_type()},\n" \
              f"device_id: {self._device_info.get_device_id()}."
        return res
