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
import os

from ._checkparam import check_isinstance, check_list_of_element, check_input_shape
from .lib import _c_lite_wrapper

__all__ = ['Context', 'DeviceInfo', 'CPUDeviceInfo', 'GPUDeviceInfo', 'AscendDeviceInfo']


class Context:
    """
    Context is used to store environment variables during execution.

    Args:
        thread_num (int, optional): Set the number of threads at runtime.
        thread_affinity_mode (int, optional): Set the thread affinity to CPU cores.
                                              0: no affinities, 1: big cores first, 2: little cores first
        thread_affinity_core_list (list[int], optional): Set the thread lists to CPU cores.
        enable_parallel (bool, optional): Set the status whether to perform model inference or training in parallel.

    Raises:
        TypeError: type of input parameters are invalid.
        ValueError: value of input parameters are invalid.

    Examples:
        >>> import mindspore_lite as mslite
        >>> context = mslite.Context(thread_num=1, thread_afffinity_mode=1, thread_affinity_core_list=[1, 2], \
        ...                          enable_parallel=False)
        >>> print(context)
        thread_num: 1, thread_affinity_mode: 1, thread_affinity_core_list: [1, 2], enable_parallel: False, \
        device_list: 0, .
    """

    def __init__(self, thread_num=None, thread_affinity_mode=None, thread_affinity_core_list=None, \
                 enable_parallel=False):
        if thread_num is not None:
            check_isinstance("thread_num", thread_num, int)
            if thread_num < 0:
                raise ValueError(f"Context's init failed, thread_num must be positive.")
        if thread_affinity_mode is not None:
            check_isinstance("thread_affinity_mode", thread_affinity_mode, int)
        check_list_of_element("thread_affinity_core_list", thread_affinity_core_list, int, enable_none=True)
        check_isinstance("enable_parallel", enable_parallel, bool)
        core_list = [] if thread_affinity_core_list is None else thread_affinity_core_list
        self._context = _c_lite_wrapper.ContextBind()
        if thread_num is not None:
            self._context.set_thread_num(thread_num)
        if thread_affinity_mode is not None:
            self._context.set_thread_affinity_mode(thread_affinity_mode)
        self._context.set_thread_affinity_core_list(core_list)
        self._context.set_enable_parallel(enable_parallel)

    def __str__(self):
        res = f"thread_num: {self._context.get_thread_num()}, " \
              f"thread_affinity_mode: {self._context.get_thread_affinity_mode()}, " \
              f"thread_affinity_core_list: {self._context.get_thread_affinity_core_list()}, " \
              f"enable_parallel: {self._context.get_enable_parallel()}, " \
              f"device_list: {self._context.get_device_list()}."
        return res

    def append_device_info(self, device_info):
        """
        Append one user-defined device info to the context

        Args:
            device_info (Union[CPUDeviceInfo, GPUDeviceInfo, AscendDeviceInfo]): device info.

        Raises:
            TypeError: type of input parameters are invalid.

        Examples:
            >>> import mindspore_lite as mslite
            >>> context = mslite.Context()
            >>> context.append_device_info(mslite.CPUDeviceInfo())
            >>> print(context)
            thread_num: 2, thread_affinity_mode: 0, thread_affinity_core_list: [], enable_parallel: False, \
            device_list: 0, .
        """
        if not isinstance(device_info, DeviceInfo):
            raise TypeError("device_info must be CPUDeviceInfo, GPUDeviceInfo or AscendDeviceInfo, but got {}.".format(
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
    Helper class to set cpu device info.

    Args:
        enable_fp16(bool, optional): enables to perform the float16 inference.

    Raises:
        TypeError: type of input parameters are invalid.

    Examples:
        >>> import mindspore_lite as mslite
        >>> cpu_device_info = mslite.CPUDeviceInfo(enable_fp16=True)
        >>> print(cpu_device_info)
        device_type: DeviceType.kCPU, enable_fp16: True.
        >>> context = mslite.Context()
        >>> context.append_device_info(cpu_device_info)
    """

    def __init__(self, enable_fp16=False):
        super(CPUDeviceInfo, self).__init__()
        check_isinstance("enable_fp16", enable_fp16, bool)
        self._device_info = _c_lite_wrapper.CPUDeviceInfoBind()
        self._device_info.set_enable_fp16(enable_fp16)

    def __str__(self):
        res = f"device_type: {self._device_info.get_device_type()}, " \
              f"enable_fp16: {self._device_info.get_enable_fp16()}."
        return res


class GPUDeviceInfo(DeviceInfo):
    """
    Helper class to set gpu device info.

    Args:
        device_id(int, optional): The device id.
        enable_fp16(bool, optional): enables to perform the float16 inference.

    Raises:
        TypeError: type of input parameters are invalid.
        ValueError: value of input parameters are invalid.

    Examples:
        >>> import mindspore_lite as mslite
        >>> gpu_device_info = mslite.GPUDeviceInfo(device_id=1, enable_fp16=False)
        >>> print(gpu_device_info)
        device_type: DeviceType.kGPU, device_id: 1, enable_fp16: False.
        >>> cpu_device_info = mslite.CPUDeviceInfo(enable_fp16=False)
        >>> context = mslite.Context()
        >>> context.append_device_info(mslite.CPUDeviceInfo(gpu_device_info))
        >>> context.append_device_info(mslite.CPUDeviceInfo(cpu_device_info))
        >>> print(context)
        thread_num: 2, thread_affinity_mode: 0, thread_affinity_core_list: [], enable_parallel: False, \
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
        res = f"device_type: {self._device_info.get_device_type()}, " \
              f"device_id: {self._device_info.get_device_id()}, " \
              f"enable_fp16: {self._device_info.get_enable_fp16()}."
        return res

    def get_rank_id(self):
        """
        Get rank id from context.

        Returns:
            int, the rank id of the context.

        Examples:
            >>> import mindspore_lite as mslite
            >>> device_info = mslite.GPUDeviceInfo(device_id=1, enable_fp16=True)
            >>> rank_id = device_info.get_rank_id()
            >>> print(rank_id)
            1
        """
        return self._device_info.get_rank_id()

    def get_group_size(self):
        """
        Get group size from context.

        Returns:
            int, the group size of the context.

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
    Helper class to set Ascend device infos.

    Args:
        device_id(int, optional): The device id.
        input_format (str, optional): Manually specify the model input format, the value can be "NCHW", "NHWC", etc.
        input_shape (dict{int:list[int]}, optional): Set shape of model inputs. e.g. {1: [1, 2, 3, 4], 2: [4, 3, 2, 1]}.
        precision_mode (str, optional): Model precision mode, the value can be "force_fp16", "allow_fp32_to_fp16",
            "must_keep_origin_dtype" or "allow_mix_precision". Default: "force_fp16".
        op_select_impl_mode (str, optional): The operator selection mode, the value can be "high_performance" or
            "high_precision". Default: "high_performance".
        dynamic_batch_size (list[int], optional): the dynamic image size of model inputs. e.g. [2, 4]
        dynamic_image_size (str, optional): image size hw e.g. "66,88;32,64" means h1:66,w1:88; h2:32,w2:64.
        fusion_switch_config_path (str, optional): Configuration file path of the convergence rule, including graph
             convergence and UB convergence. The system has built-in graph convergence and UB convergence rules, which
             are enableed by default. You can disable the reuls specified in the file by setting this parameter.
        insert_op_cfg_path (str, optional): Path of aipp config file.

    Raises:
        TypeError: type of input parameters are invalid.
        ValueError: value of input parameters are invalid.
        RuntimeError: file path does not exist


    Examples:
        >>> import mindspore_lite as mslite
        >>> ascend_device_info = mslite.AscendDeviceInfo(device_id=0, input_format="NCHW", \
        ...                      input_shape={1: [1, 3, 28, 28]}, precision_mode="force_fp16", \
        ...                      op_select_impl_mode="high_performance", dynamic_batch_size=None, \
        ...                      dynamic_image_size="", fusion_switch_config_path="", insert_op_cfg_path="")
        >>> print(ascend_device_info)
        >>> cpu_device_info = mslite.CPUDeviceInfo(enable_fp16=False)
        >>> context = mslite.Context()
        >>> context.append_device_info(mslite.CPUDeviceInfo(gpu_device_info))
        >>> context.append_device_info(mslite.CPUDeviceInfo(ascend_device_info))
        >>> print(context)
        thread_num: 2, thread_affinity_mode: 0, thread_affinity_core_list: [], enable_parallel: False, \
        device_list: 3, 0, .
    """

    def __init__(self, device_id=0, input_format=None, input_shape=None, precision_mode="force_fp16",
                 op_select_impl_mode="high_performance", dynamic_batch_size=None, dynamic_image_size="",
                 fusion_switch_config_path="", insert_op_cfg_path=""):
        super(AscendDeviceInfo, self).__init__()
        check_isinstance("device_id", device_id, int)
        check_isinstance("input_format", input_format, str, enable_none=True)
        check_input_shape("input_shape", input_shape, enable_none=True)
        check_isinstance("precision_mode", precision_mode, str)
        check_isinstance("op_select_impl_mode", op_select_impl_mode, str)
        check_list_of_element("dynamic_batch_size", dynamic_batch_size, int, enable_none=True)
        check_isinstance("dynamic_image_size", dynamic_image_size, str)
        check_isinstance("fusion_switch_config_path", fusion_switch_config_path, str)
        check_isinstance("insert_op_cfg_path", insert_op_cfg_path, str)
        if device_id < 0:
            raise ValueError(f"AscendDeviceInfo's init failed, device_id must be positive.")
        if fusion_switch_config_path != "":
            if not os.path.exists(fusion_switch_config_path):
                raise RuntimeError(f"AscendDeviceInfo's init failed, fusion_switch_config_path does not exist!")
        if insert_op_cfg_path != "":
            if not os.path.exists(insert_op_cfg_path):
                raise RuntimeError(f"AscendDeviceInfo's init failed, insert_op_cfg_path does not exist!")
        input_format_list = "" if input_format is None else input_format
        input_shape_list = {} if input_shape is None else input_shape
        batch_size_list = [] if dynamic_batch_size is None else dynamic_batch_size
        self._device_info = _c_lite_wrapper.AscendDeviceInfoBind()
        self._device_info.set_device_id(device_id)
        self._device_info.set_input_format(input_format_list)
        self._device_info.set_input_shape(input_shape_list)
        self._device_info.set_precision_mode(precision_mode)
        self._device_info.set_op_select_impl_mode(op_select_impl_mode)
        self._device_info.set_dynamic_batch_size(batch_size_list)
        self._device_info.set_dynamic_image_size(dynamic_image_size)
        self._device_info.set_fusion_switch_config_path(fusion_switch_config_path)
        self._device_info.set_insert_op_cfg_path(insert_op_cfg_path)

    def __str__(self):
        res = f"device_type: {self._device_info.get_device_type()}, " \
              f"device_id: {self._device_info.get_device_id()}, " \
              f"input_format: {self._device_info.get_input_format()}, " \
              f"input_shape: {self._device_info.get_input_shape()}, " \
              f"precision_mode: {self._device_info.get_precision_mode()}, " \
              f"op_select_impl_mode: {self._device_info.get_op_select_impl_mode()}, " \
              f"dynamic_batch_size: {self._device_info.get_dynamic_batch_size()}, " \
              f"dynamic_image_size: {self._device_info.get_dynamic_image_size()}, " \
              f"fusion_switch_config_path: {self._device_info.get_fusion_switch_config_path()}, " \
              f"insert_op_cfg_path: {self._device_info.get_insert_op_cfg_path()}."
        return res
