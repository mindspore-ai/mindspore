# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
from __future__ import absolute_import
import os

from mindspore_lite._checkparam import check_isinstance, check_list_of_element
from mindspore_lite.lib import _c_lite_wrapper
from mindspore_lite._check_ascend import check_ascend_env

__all__ = ['Context']


class Context:
    """
    The `Context` class is used to transfer environment variables during execution.

    The context should be configured before running the program.
    If it is not configured, the `target` will be set to ``cpu``, and automatically set ``cpu`` attributes by default.

    Context.parallel defines the context and configuration of `ModelParallelRunner` class.

    Context.parallel properties:
        - **workers_num** (int) - the num of workers. A `ModelParallelRunner` contains multiple workers, which
          are the units that actually perform parallel inferring. Setting `workers_num` to 0 represents
          `workers_num` will be automatically adjusted based on computer performance and core numbers.
        - **config_info** (dict{str, dict{str, str}}) - Nested map for transferring user defined options during building
          `ModelParallelRunner` online. More configurable options refer to `config_path` .
          For example, ``{"model_file": {"mindir_path": "/home/user/model_graph.mindir"}}``.
          `section` is ``"model_file"``, one of the keys is ``"mindir_path"``,
          the corresponding value in the map is ``"/home/user/model_graph.mindir"``.
        - **config_path** (str) - Set the config file path. The config file is used to transfer user-defined
          options during building `ModelParallelRunner` . In the following scenarios, users may need to set the
          parameter. For example, ``"/home/user/config.txt"``.

          - Usage 1: Set mixed precision inference. The content and description of the configuration file are as
            follows:

            .. code-block::

                [execution_plan]
                [op_name1]=data_Type: float16 (The operator named op_name1 sets the data type as float16)
                [op_name2]=data_Type: float32 (The operator named op_name2 sets the data type as float32)

          - Usage 2: When GPU inference, set the configuration of TensorRT. The content and description of the
            configuration file are as follows:

            .. code-block::

                [ms_cache]
                serialize_Path=[serialization model path](storage path of serialization model)
                [gpu_context]
                input_shape=input_Name: [input_dim] (Model input dimension, for dynamic shape)
                dynamic_Dims=[min_dim~max_dim] (dynamic dimension range of model input, for dynamic shape)
                opt_Dims=[opt_dim] (the optimal input dimension of the model, for dynamic shape)

          - Usage 3: For the large model, when using the model buffer to load and compile, you need to set the path
            of the weight file separately through passing the path of the large model. And it is necessary to ensure
            that the large model file and the folder where the weight file is located are in the same folder.
            For example, when the directory is as follows:

            .. code-block::

                .
                └── /home/user/
                     ├── model_graph.mindir
                     └── model_variables
                          └── data_0

            The content and description of the configuration file are as follows:

            .. code-block::

                [model_file]
                mindir_path=[/home/user/model_graph.mindir](storage path of the large model)

    Examples:
        >>> # create default context, which target is cpu by default.
        >>> import mindspore_lite as mslite
        >>> context = mslite.Context()
        >>> print(context)
        target: ['cpu'].
        >>> # testcase 2 about context's attribute parallel based on server inference package
        >>> # (export MSLITE_ENABLE_SERVER_INFERENCE=on before compile lite or use cloud inference package)
        >>> import mindspore_lite as mslite
        >>> context = mslite.Context()
        >>> context.target = ["cpu"]
        >>> context.parallel.workers_num = 4
        >>> context.parallel.config_info = {"model_file": {"mindir_path": "/home/user/model_graph.mindir"}}
        >>> context.parallel.config_path = "/home/user/config.txt"
        >>> print(context.parallel)
        workers num: 4,
        config info: model_file: mindir_path /home/user/model_graph.mindir,
        config path: /home/user/config.txt.
    """

    def __init__(self):
        self._context = _InnerContext()
        self.cpu = _CPU(self._context)
        self.gpu = _GPU()
        self.ascend = _Ascend()
        self.target = ["cpu"]
        if hasattr(_c_lite_wrapper, "RunnerConfigBind"):
            self.parallel = _Parallel(self._context)

    def __str__(self):
        res = f"target: {self.target}."
        return res

    @property
    def target(self):
        """
        Get the target device information of context.

        Currently support target: ``"cpu"`` , ``"gpu"`` , ``"ascend"``.

        Note:
            After gpu is added to target, cpu will be added automatically as the backup target.
            Because when ops are not supported on gpu, The system will try whether the cpu supports it.
            At that time, need to switch to the context with cpu.

            After Ascend is added, cpu will be added automatically as the backup target. when the inputs format of the
            original model is inconsistent with that of the model generated by Converter, the model generated by
            Converter on Ascend device will contain the 'Transpose' node, which needs to be executed on the cpu device
            currently. So it needs to switch to the context with cpu target.

        cpu properties:
            - **inter_op_parallel_num** (int) - Set the parallel number of operators at runtime.
              `inter_op_parallel_num` cannot be greater than `thread_num` . Setting `inter_op_parallel_num`
              to ``0`` represents `inter_op_parallel_num` will be automatically adjusted based on computer
              performance and core num.
            - **precision_mode** (str) - Set the mix precision mode. Options are ``"preferred_fp16"`` ,
              ``"enforce_fp32"``.

              - ``"preferred_fp16"`` : prefer to use fp16.
              - ``"enforce_fp32"`` : force use fp32.

            - **thread_num** (int) - Set the number of threads at runtime. `thread_num` cannot be less than
              `inter_op_parallel_num` . Setting `thread_num` to 0 represents `thread_num` will be automatically
              adjusted based on computer performance and core numbers.
            - **thread_affinity_mode** (int) - Set the mode of the CPU core binding policy at runtime. The
              following `thread_affinity_mode` are supported.

              - ``0`` : no binding core.
              - ``1`` : binding big cores first.
              - ``2`` : binding middle cores first.

            - **thread_affinity_core_list** (list[int]) - Set the list of CPU core binding policies at runtime.
              For example, [0,1] represents the specified binding of CPU0 and CPU1.

        gpu properties:
            - **device_id** (int) - The device id.
            - **group_size** (int) - the number of the clusters. Get only, not settable.
            - **precision_mode** (str) - Set the mix precision mode. Options are ``"preferred_fp16"`` ,
              ``"enforce_fp32"``.

              - ``"preferred_fp16"``: prefer to use fp16.
              - ``"enforce_fp32"``: force use fp32.

            - **rank_id** (int) - the ID of the current device in the cluster, which starts from 0. Get only,
              not settable.

        ascend properties:
            - **device_id** (int) - The device id.
            - **precision_mode** (str) - Set the mix precision mode. Options are ``"enforce_fp32"`` ,
              ``"preferred_fp32"`` , ``"enforce_fp16"`` , ``"enforce_origin"`` , ``"preferred_optimal"``.

              - ``"enforce_fp32"``: ACL option is force_fp32, force use fp32.
              - ``"preferred_fp32"``: ACL option is allow_fp32_to_fp16, prefer to use fp32.
              - ``"enforce_fp16"``: ACL option is force_fp16, force use fp16.
              - ``"enforce_origin"``: ACL option is must_keep_origin_dtype, force use original type.
              - ``"preferred_optimal"``: ACL option is allow_mix_precision, prefer to use fp16+ mix precision mode.

            - **provider** (str) - The provider that supports the inference capability of the target device,
              can be ``""`` or ``"ge"``. The default is ``""``.
            - **rank_id** (int) - The ID of the current device in the cluster, which starts from ``0``.

        Returns:
            list[str], the target device information of context.

        Examples:
            >>> # create default context, which target is cpu by default.
            >>> import mindspore_lite as mslite
            >>> context = mslite.Context()
            >>> # set context with cpu target.
            >>> context.target = ["cpu"]
            >>> print(context.target)
            ['cpu']
            >>> context.cpu.precision_mode = "preferred_fp16"
            >>> context.cpu.thread_num = 2
            >>> context.cpu.inter_op_parallel_num = 2
            >>> context.cpu.thread_affinity_mode = 1
            >>> context.cpu.thread_affinity_core_list = [0,1]
            >>> print(context.cpu)
            device_type: DeviceType.kCPU,
            precision_mode: preferred_fp16,
            thread_num: 2,
            inter_op_parallel_num: 2,
            thread_affinity_mode: 1,
            thread_affinity_core_list: [0, 1].
            >>> # set context with gpu target.
            >>> context.target = ["gpu"]
            >>> print(context.target)
            ['gpu']
            >>> context.gpu.precision_mode = "preferred_fp16"
            >>> context.gpu.device_id = 2
            >>> print(context.gpu.rank_id)
            0
            >>> print(context.gpu.group_size)
            1
            >>> print(context.gpu)
            device_type: DeviceType.kGPU,
            precision_mode: preferred_fp16,
            device_id: 2,
            rank_id: 0,
            group_size: 1.
            >>> # set context with ascend target.
            >>> context.target = ["ascend"]
            >>> print(context.target)
            ['ascend']
            >>> context.ascend.precision_mode = "enforce_fp32"
            >>> context.ascend.device_id = 2
            >>> context.ascend.provider = "ge"
            >>> context.ascend.rank_id = 0
            >>> print(context.ascend)
            device_type: DeviceType.kAscend,
            precision_mode: enforce_fp32,
            device_id: 2,
            provider: ge,
            rank_id: 0.
        """
        return self._target

    @target.setter
    def target(self, target):
        """
        Set the target device information of context.

        Args:
            target (list[str]): the target device information of context.
            Currently support target: ["cpu"] | ["gpu"] | ["ascend"].

        Raises:
            TypeError: `target` is not a list.
            TypeError: `target` is a list, but the elements are not str.
            ValueError: `target` is a list, but the elements are not in ['cpu', 'gpu', 'ascend'].
        """
        target = ["cpu"] if not target else target
        check_list_of_element("target", target, str)
        for ele in target:
            if ele.lower() not in ["cpu", "gpu", "ascend"]:
                raise ValueError(f"target elements must be in ['cpu', 'gpu', 'ascend'], but got {ele.lower()}.")
        self._context.clear_target()
        need_cpu_backup = False
        for ele in target:
            if ele.lower() == "ascend":
                check_ascend_env()

                self._context.append_device_info(self.ascend)
                need_cpu_backup = True
            elif ele.lower() == "gpu":
                self._context.append_device_info(self.gpu)
                need_cpu_backup = True
            else:
                self._context.append_device_info(self.cpu)
        if need_cpu_backup:
            self._context.append_device_info(self.cpu)
        self._target = target

    @property
    def group_info_file(self):
        """Get or set communication group info file for distributed inference.

        In the pipeline parallel scenario, different stage device nodes are in different communication groups. When
        exporting the model, set the `group_ckpt_save_file` parameter in interface
        [mindspore.set_auto_parallel_context](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/mindspore/mindspore.set_auto_parallel_context.html)
        to export the group file information. In addition, in non pipeline parallel scenarios, if there
        are communication operators involving local communication groups, the group file information also needs to be
        exported through the 'group_ckpt_save_file' parameter.

        Examples:
            >>> # export communication group information file when export mindir
            >>> import mindspore
            >>> mindspore.set_auto_parallel_context(group_ckpt_save_file=f"{export_dir}/group_config_{rank_id}.pb")
            >>>
            >>> # use communication group information file when load mindir
            >>> import mindspore_lite as mslite
            >>> context = mslite.Context()
            >>> context.group_info_file = f"{export_dir}/group_config_{rank_id}.pb"
        """
        return self._context.group_info_file

    @group_info_file.setter
    def group_info_file(self, group_info_file):
        """Set communication group information for distributed inference."""
        check_isinstance("group_info_file", group_info_file, str)
        self._context.group_info_file = group_info_file


class _InnerContext:
    """_InnerContext is used to bind Python API(Context) to C++ API(Context)."""
    def __init__(self):
        self._inner_context = _c_lite_wrapper.ContextBind()

    @property
    def cpu_thread_num(self):
        """Get the number of threads at runtime."""
        return self._inner_context.get_thread_num()

    @cpu_thread_num.setter
    def cpu_thread_num(self, cpu_thread_num):
        """Set the number of threads at runtime."""
        check_isinstance("cpu_thread_num", cpu_thread_num, int)
        if cpu_thread_num < 0:
            raise ValueError(f"cpu_thread_num must be a non-negative int.")
        self._inner_context.set_thread_num(cpu_thread_num)

    @property
    def cpu_inter_op_parallel_num(self):
        """Get the parallel number of operators at runtime."""
        return self._inner_context.get_inter_op_parallel_num()

    @cpu_inter_op_parallel_num.setter
    def cpu_inter_op_parallel_num(self, cpu_inter_op_parallel_num):
        """Set the parallel number of operators at runtime."""
        check_isinstance("cpu_inter_op_parallel_num", cpu_inter_op_parallel_num, int)
        if cpu_inter_op_parallel_num < 0:
            raise ValueError(f"Context's init failed, cpu_inter_op_parallel_num must be a non-negative int.")
        self._inner_context.set_inter_op_parallel_num(cpu_inter_op_parallel_num)

    @property
    def cpu_thread_affinity_mode(self):
        """Get the mode of the CPU core binding policy at runtime."""
        return self._inner_context.get_thread_affinity_mode()

    @cpu_thread_affinity_mode.setter
    def cpu_thread_affinity_mode(self, cpu_thread_affinity_mode):
        """Set the mode of the CPU core binding policy at runtime."""
        check_isinstance("cpu_thread_affinity_mode", cpu_thread_affinity_mode, int)
        self._inner_context.set_thread_affinity_mode(cpu_thread_affinity_mode)

    @property
    def cpu_thread_affinity_core_list(self):
        """Get the list of CPU core binding policies at runtime."""
        return self._inner_context.get_thread_affinity_core_list()

    @cpu_thread_affinity_core_list.setter
    def cpu_thread_affinity_core_list(self, cpu_thread_affinity_core_list):
        """Set the list of CPU core binding policies at runtime."""
        check_list_of_element("cpu_thread_affinity_core_list", cpu_thread_affinity_core_list, int, enable_none=False)
        self._inner_context.set_thread_affinity_core_list(cpu_thread_affinity_core_list)

    def get_target(self):
        """Get the target device information of context."""
        return self._inner_context.get_device_list()

    def clear_target(self):
        """Clear the target device information of context."""
        self._inner_context.clear_device_info()

    def append_device_info(self, target):
        """Append one user-defined target device info to the context."""
        check_isinstance("target", target, _Target)
        self._inner_context.append_device_info(target._device_info)

    @property
    def group_info_file(self):
        """Get communication group info file for distributed inference."""
        return self._inner_context.get_group_info_file()

    @group_info_file.setter
    def group_info_file(self, group_info_file):
        """Set communication group info file for distributed inference."""
        check_isinstance("group_info_file", group_info_file, str)
        self._inner_context.set_group_info_file(group_info_file)


class _Target:
    """
    Helper class used to describe device hardware information.
    """

    def __init__(self):
        """ Initialize _Target"""


class _CPU(_Target):
    """
    Helper class used to describe CPU device hardware information, and it inherits :class:`mindspore_lite._Target`
    base class.

    Args:
        inner_context(_InnerContext): Use to set inner context's cpu parameters.
    """

    def __init__(self, inner_context):
        super(_CPU, self).__init__()
        check_isinstance("inner_context", inner_context, _InnerContext)
        self._inner_context = inner_context
        self._device_info = _c_lite_wrapper.CPUDeviceInfoBind()

    def __str__(self):
        res = f"device_type: {self._device_info.get_device_type()},\n" \
              f"precision_mode: {self.precision_mode},\n" \
              f"thread_num: {self.thread_num},\n" \
              f"inter_op_parallel_num: {self.inter_op_parallel_num},\n" \
              f"thread_affinity_mode: {self.thread_affinity_mode},\n" \
              f"thread_affinity_core_list: {self.thread_affinity_core_list}."
        return res

    @property
    def precision_mode(self):
        """Get mixed precision mode."""
        if self._device_info.get_enable_fp16():
            return "preferred_fp16"
        return "enforce_fp32"

    @precision_mode.setter
    def precision_mode(self, cpu_precision_mode):
        """
        Set mixed precision mode.

        Args:
            cpu_precision_mode (str): Set mixed precision mode. CPU options are "preferred_fp16" | "enforce_fp32".

                - "preferred_fp16": prefer to use fp16.
                - "enforce_fp32": force use fp32.

        Raises:
            TypeError: `cpu_precision_mode` is not a str.
            ValueError: `cpu_precision_mode` is neither "enforce_fp32" nor "preferred_fp16" when it is a str.
        """
        check_isinstance("cpu_precision_mode", cpu_precision_mode, str)
        if cpu_precision_mode not in ["enforce_fp32", "preferred_fp16"]:
            raise ValueError(f"cpu_precision_mode must be in [enforce_fp32, preferred_fp16],"
                             f" but got {cpu_precision_mode}.")
        if cpu_precision_mode == "preferred_fp16":
            self._device_info.set_enable_fp16(True)
        else:
            self._device_info.set_enable_fp16(False)

    @property
    def thread_num(self):
        """Get the number of threads at runtime."""
        return self._inner_context.cpu_thread_num

    @thread_num.setter
    def thread_num(self, cpu_thread_num):
        """
        Set the number of threads at runtime.

        Args:
            cpu_thread_num (int): Set the number of threads at runtime. `cpu_thread_num` cannot be less than
                `cpu_inter_op_parallel_num` . Setting `cpu_thread_num` to 0 represents `cpu_thread_num` will be
                automatically adjusted based on computer performance and core numbers.

        Raises:
            TypeError: `cpu_thread_num` is not an int.
            ValueError: `cpu_thread_num` is less than 0.
        """
        check_isinstance("cpu_thread_num", cpu_thread_num, int)
        if cpu_thread_num < 0:
            raise ValueError(f"cpu_thread_num must be a non-negative int.")
        self._inner_context.cpu_thread_num = cpu_thread_num

    @property
    def inter_op_parallel_num(self):
        """Get the parallel number of operators at runtime."""
        return self._inner_context.cpu_inter_op_parallel_num

    @inter_op_parallel_num.setter
    def inter_op_parallel_num(self, cpu_inter_op_parallel_num):
        """
        Set the parallel number of operators at runtime.

        Args:
            cpu_inter_op_parallel_num (int): Set the parallel number of operators at runtime.
                `cpu_inter_op_parallel_num` cannot be greater than `cpu_thread_num` . Setting
                `cpu_inter_op_parallel_num` to 0 represents `cpu_inter_op_parallel_num` will be automatically adjusted
                based on computer performance and core num.

        Raises:
            TypeError: `cpu_inter_op_parallel_num` is not an int.
            ValueError: `cpu_inter_op_parallel_num` is less than 0.
        """
        check_isinstance("cpu_inter_op_parallel_num", cpu_inter_op_parallel_num, int)
        if cpu_inter_op_parallel_num < 0:
            raise ValueError(f"cpu_inter_op_parallel_num must be a non-negative int.")
        self._inner_context.cpu_inter_op_parallel_num = cpu_inter_op_parallel_num

    @property
    def thread_affinity_mode(self):
        """Get the mode of the CPU core binding policy at runtime."""
        return self._inner_context.cpu_thread_affinity_mode

    @thread_affinity_mode.setter
    def thread_affinity_mode(self, cpu_thread_affinity_mode):
        """
        Set the mode of the CPU core binding policy at runtime.

        Args:
            cpu_thread_affinity_mode (int): Set the mode of the CPU core binding policy at runtime. The
                following `cpu_thread_affinity_mode` are supported.

                - 0: no binding core.
                - 1: binding big cores first.
                - 2: binding middle cores first.

        Raises:
            TypeError: `cpu_thread_affinity_mode` is not an int.
        """
        check_isinstance("cpu_thread_affinity_mode", cpu_thread_affinity_mode, int)
        self._inner_context.cpu_thread_affinity_mode = cpu_thread_affinity_mode

    @property
    def thread_affinity_core_list(self):
        """Get the list of CPU core binding policies at runtime."""
        return self._inner_context.cpu_thread_affinity_core_list

    @thread_affinity_core_list.setter
    def thread_affinity_core_list(self, cpu_thread_affinity_core_list):
        """
        Set the list of CPU core binding policies at runtime.

        Args:
            cpu_thread_affinity_core_list (list[int]): Set the list of CPU core binding policies at runtime.
                For example, [0,1] represents the specified binding of CPU0 and CPU1.

        Raises:
            TypeError: `cpu_thread_affinity_core_list` is not a list.
            TypeError: `cpu_thread_affinity_core_list` is a list, but the elements are not int.
        """
        check_list_of_element("cpu_thread_affinity_core_list", cpu_thread_affinity_core_list, int, enable_none=False)
        self._inner_context.cpu_thread_affinity_core_list = cpu_thread_affinity_core_list


class _GPU(_Target):
    """
    Helper class used to describe GPU device hardware information, and it inherits :class:`mindspore_lite._Target`
    base class.
    """

    def __init__(self):
        super(_GPU, self).__init__()
        self._device_info = _c_lite_wrapper.GPUDeviceInfoBind()

    def __str__(self):
        res = f"device_type: {self._device_info.get_device_type()},\n" \
              f"precision_mode: {self.precision_mode},\n" \
              f"device_id: {self.device_id},\n" \
              f"rank_id: {self.rank_id},\n" \
              f"group_size: {self.group_size}."
        return res

    @property
    def precision_mode(self):
        """Get mixed precision mode."""
        if self._device_info.get_enable_fp16():
            return "preferred_fp16"
        return "enforce_fp32"

    @precision_mode.setter
    def precision_mode(self, gpu_precision_mode):
        """
        Set mixed precision mode.

        Args:
            gpu_precision_mode (str): Set mixed precision mode. GPU options are "preferred_fp16" | "enforce_fp32".

                - "preferred_fp16": prefer to use fp16.
                - "enforce_fp32": force use fp32.

        Raises:
            TypeError: `gpu_precision_mode` is not a str.
            ValueError: `gpu_precision_mode` is neither "enforce_fp32" nor "preferred_fp16" when it is a str.
        """
        check_isinstance("gpu_precision_mode", gpu_precision_mode, str)
        if gpu_precision_mode not in ["enforce_fp32", "preferred_fp16"]:
            raise ValueError(f"gpu_precision_mode must be in [enforce_fp32, preferred_fp16],"
                             f" but got {gpu_precision_mode}.")
        if gpu_precision_mode == "preferred_fp16":
            self._device_info.set_enable_fp16(True)
        else:
            self._device_info.set_enable_fp16(False)

    @property
    def device_id(self):
        """Get the device id."""
        return self._device_info.get_device_id()

    @device_id.setter
    def device_id(self, gpu_device_id):
        """
        Set the device id.

        Args:
            gpu_device_id(int): The device id.

        Raises:
            TypeError: `gpu_device_id` is not an int.
            ValueError: `gpu_device_id` is less than 0.
        """
        check_isinstance("gpu_device_id", gpu_device_id, int)
        if gpu_device_id < 0:
            raise ValueError(f"gpu_device_id must be a non-negative int.")
        self._device_info.set_device_id(gpu_device_id)

    @property
    def rank_id(self):
        """
        Get the ID of the current device in the cluster from context.

        Returns:
            int, the ID of the current device in the cluster, which starts from 0.
        """
        return self._device_info.get_rank_id()

    @property
    def group_size(self):
        """
        Get the number of the clusters from context.

        Returns:
            int, the number of the clusters.
        """
        return self._device_info.get_group_size()


class _Ascend(_Target):
    """
    Helper class used to describe Ascend device hardware information, and it inherits :class:`mindspore_lite._Target`
    base class.
    """

    def __init__(self):
        super(_Ascend, self).__init__()
        self._device_info = _c_lite_wrapper.AscendDeviceInfoBind()

    def __str__(self):
        res = f"device_type: {self._device_info.get_device_type()},\n" \
              f"precision_mode: {self.precision_mode},\n" \
              f"device_id: {self.device_id}." \
              f"provider: {self.provider}." \
              f"rank_id: {self.rank_id}."
        return res

    @property
    def precision_mode(self):
        """Get mixed precision mode."""
        return self._device_info.get_precision_mode()

    @precision_mode.setter
    def precision_mode(self, ascend_precision_mode):
        """
        Set mixed precision mode.

        Args:
            ascend_precision_mode (str): Set mixed precision mode. Ascend options are "enforce_fp32" |
                "preferred_fp32" | "enforce_fp16" | "enforce_origin" | "preferred_optimal".

              - "enforce_fp32": ACL option is force_fp32, force use fp32.
              - "preferred_fp32": ACL option is allow_fp32_to_fp16, prefer to use fp32.
              - "enforce_fp16": ACL option is force_fp16, force use fp16.
              - "enforce_origin": ACL option is must_keep_origin_dtype, force use original type.
              - "preferred_optimal": ACL option is allow_mix_precision, prefer to use fp16+ mix precision mode.

        Raises:
            TypeError: `ascend_precision_mode` is not a str.
            ValueError: `ascend_precision_mode` is not in ["enforce_fp32", "preferred_fp32", "enforce_fp16",
            "enforce_origin", "preferred_optimal"] when it is a str.
        """
        check_isinstance("ascend_precision_mode", ascend_precision_mode, str)
        if ascend_precision_mode not in ["enforce_fp32", "preferred_fp32", "enforce_fp16", "enforce_origin",
                                         "preferred_optimal"]:
            raise ValueError(f"ascend_precision_mode must be in [enforce_fp32, preferred_fp32, enforce_fp16, "
                             f"enforce_origin, preferred_optimal], but got {ascend_precision_mode}.")
        self._device_info.set_precision_mode(ascend_precision_mode)

    @property
    def device_id(self):
        """Get the device id."""
        return self._device_info.get_device_id()

    @device_id.setter
    def device_id(self, ascend_device_id):
        """
        Set the device id.

        Args:
            ascend_device_id(int): The device id.

        Raises:
            TypeError: `ascend_device_id` is not an int.
            ValueError: `ascend_device_id` is less than 0.
        """
        check_isinstance("ascend_device_id", ascend_device_id, int)
        if ascend_device_id < 0:
            raise ValueError(f"ascend_device_id must be a non-negative int.")
        self._device_info.set_device_id(ascend_device_id)

    @property
    def rank_id(self):
        """
        Get the ID of the current device in the cluster from context.

        Returns:
            int, the ID of the current device in the cluster, which starts from 0.
        """
        return self._device_info.get_rank_id()

    @rank_id.setter
    def rank_id(self, ascend_rank_id):
        """
        Set the ID of the current device in the cluster from context.

        Args:
            ascend_rank_id(int): The rank id.

        Raises:
            TypeError: `ascend_rank_id` is not an int.
            ValueError: `ascend_rank_id` is less than 0.
        """
        check_isinstance("ascend_rank_id", ascend_rank_id, int)
        if ascend_rank_id < 0:
            raise ValueError(f"ascend_rank_id must be a non-negative int.")
        self._device_info.set_rank_id(ascend_rank_id)

    @property
    def provider(self):
        """Get the provider that supports the inference capability of target device."""
        return self._device_info.get_provider()

    @provider.setter
    def provider(self, ascend_provider):
        """
        Set the provider that supports the inference capability of target device.

        Args:
            ascend_provider(str): The ascend provider, which can be "" or "ge", default "".

        Raises:
            TypeError: `ascend_provider` is not a str.
        """
        check_isinstance("ascend_provider", ascend_provider, str)
        self._device_info.set_provider(ascend_provider)


class _Parallel:
    """
    _Parallel Class defines the context and configuration of `ModelParallelRunner` class.

    Args:
        context (Context, optional): Define the context used to store options during execution. Default: ``None``.

    Raises:
        TypeError: `context` is neither a Context nor None.
        RuntimeError: Not MindSpore Lite serving package, can't set parallel.
    """

    def __init__(self, context=None):
        if hasattr(_c_lite_wrapper, "RunnerConfigBind"):
            self._runner_config = _c_lite_wrapper.RunnerConfigBind()
        else:
            raise RuntimeError(f"parallel init failed, If you want to set parallel, you need to build"
                               f"MindSpore Lite serving package by export MSLITE_ENABLE_SERVER_INFERENCE=on.")
        if context is not None:
            self._runner_config.set_context(context._inner_context)

    def __str__(self):
        res = f"workers num: {self.workers_num},\n" \
              f"config info: {self.config_info},\n" \
              f"config file: {self.config_path}."
        return res

    @property
    def workers_num(self):
        """Get the num of workers."""
        return self._runner_config.get_workers_num()

    @workers_num.setter
    def workers_num(self, workers_num):
        """
        Set the num of workers.

        Args:
            workers_num (int): the num of workers. A `ModelParallelRunner` contains multiple workers, which
                are the units that actually perform parallel inferring. Setting `workers_num` to 0 represents
                `workers_num` will be automatically adjusted based on computer performance and core numbers.

        Raises:
            TypeError: `workers_num` is not an int.
            ValueError: `workers_num` is an int, but it is less than 0.
        """
        check_isinstance("workers_num", workers_num, int)
        if workers_num < 0:
            raise ValueError(f"Set parallel failed, workers_num must be a non-negative int.")
        self._runner_config.set_workers_num(workers_num)

    @property
    def config_info(self):
        """Get Nested map for transferring user defined options during building `ModelParallelRunner` online."""
        return self._runner_config.get_config_info_string().rstrip("\n")

    @config_info.setter
    def config_info(self, config_info):
        """
        set Nested map for transferring user defined options during building `ModelParallelRunner` online.

        Args:
            config_info (dict{str, dict{str, str}}): Nested map for transferring user defined options during building
                `ModelParallelRunner` online. More configurable options refer to `config_path` .
                For example, {"model_file": {"mindir_path": "/home/user/model_graph.mindir"}}.
                `section` is "model_file", value is in dict format, one of the keys is "mindir_path",
                the corresponding value is "/home/user/model_graph.mindir".

        Raises:
            TypeError: `config_info` is not a dict.
            TypeError: `config_info` is a dict, but the key is not str.
            TypeError: `config_info` is a dict, the key is str, but the value is not dict.
            TypeError: `config_info` is a dict, the key is str, the value is dict, but the key of value is not str.
            TypeError: `config_info` is a dict, the key is str, the value is dict, the key of the value is str, but
                the value of the value is not str.
        """
        check_isinstance("config_info", config_info, dict)
        for k, v in config_info.items():
            check_isinstance("config_info_key", k, str)
            check_isinstance("config_info_value", v, dict)
            for v_k, v_v in v.items():
                check_isinstance("config_info_value_key", v_k, str)
                check_isinstance("config_info_value_value", v_v, str)
        for k, v in config_info.items():
            self._runner_config.set_config_info(k, v)

    @property
    def config_path(self):
        """Get the config file path."""
        return self._runner_config.get_config_path()

    @config_path.setter
    def config_path(self, config_path):
        """
        Set the config file path.

        Args:
            config_path (str): Set the config file path. the config file is used to transfer user defined
                options during building `ModelParallelRunner` . In the following scenarios, users may need to set the
                parameter. For example, "/home/user/config.txt".

                - Usage 1: Set mixed precision inference. The content and description of the configuration file are as
                  follows:

                  .. code-block::

                      [execution_plan]
                      [op_name1]=data_Type: float16 (The operator named op_name1 sets the data type as float16)
                      [op_name2]=data_Type: float32 (The operator named op_name2 sets the data type as float32)

                - Usage 2: When GPU inference, set the configuration of TensorRT. The content and description of the
                  configuration file are as follows:

                  .. code-block::

                      [ms_cache]
                      serialize_Path=[serialization model path](storage path of serialization model)
                      [gpu_context]
                      input_shape=input_Name: [input_dim] (Model input dimension, for dynamic shape)
                      dynamic_Dims=[min_dim~max_dim] (dynamic dimension range of model input, for dynamic shape)
                      opt_Dims=[opt_dim] (the optimal input dimension of the model, for dynamic shape)

                - Usage 3: For the large model, when using the model buffer to load and compile, you need to set the
                  path of the weight file separately through passing the path of the large model. And it is necessary to
                  ensure that the large model file and the folder where the weight file is located are in the same
                  folder. For example, when the directory is as follows:

                  .. code-block::

                      .
                      └── /home/user/
                           ├── model_graph.mindir
                           └── model_variables
                                └── data_0

                  The content and description of the configuration file are as follows:

                  .. code-block::

                      [model_file]
                      mindir_path=[/home/user/model_graph.mindir](storage path of the large model)

        Raises:
            TypeError: `config_path` is not a str.
            ValueError: `config_path` does not exist.
        """
        check_isinstance("config_path", config_path, str)
        if config_path != "":
            if not os.path.exists(config_path):
                raise ValueError(f"Set parallel failed, config_path does not exist!")
            self._runner_config.set_config_path(config_path)

    @property
    def device_ids(self):
        """Get the device id list."""
        return self._runner_config.get_device_ids()

    @device_ids.setter
    def device_ids(self, device_ids):
        """
        Set the device id list.

        Args:
            device_ids(list): A `ModelParallelRunner` contains multiple workers, set the device id of each
                worker based on the device_ids sequence. If the device_ids length is less than workers_num,
                the worker will distribute evenly to each device.

        Raises:
            TypeError: `device_ids` is not a list.
            TypeError: `device_ids` is a list, but the elements are not int.
            ValueError: element of `device_ids` is less than 0.
        """
        check_list_of_element("device_ids", device_ids, int, enable_none=False)
        for _, element in enumerate(device_ids):
            if element < 0:
                raise ValueError(f"Set parallel failed, device_ids contain a negative number.")
        self._runner_config.set_device_ids(device_ids)
