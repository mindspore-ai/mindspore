# Copyright 2024 Huawei Technologies Co., Ltd
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

"""Executor manager interfaces."""
from mindspore._c_expression import RuntimeConf
from mindspore.runtime.thread_bind_core import _get_cpu_affinity_strategy, _validate_affinity_cpu_list, \
    _validate_module_cpu_index, _adapt_to_dict
from mindspore._checkparam import args_type_check
from mindspore import _checkparam as Validator
from mindspore import log as logger



def launch_blocking():
    """
    Indicates that synchronizing the execution of the startup device reduces the execution performance of the program.

    - In the initial state when this interface is not called, the operator executes asynchronously on the device.
      In this case, when an error occurs in the execution of the operator,
      it will not be possible to locate the position of the particular error script code.
    - When this interface is called, the operator is executed in a synchronized manner on the device.
      At this point, when an error occurs in the execution of the operator,
      the location of the erroneous script code can be located based on the error call stack.

    Examples:
        >>> import mindspore as ms
        >>> ms.set_device("Ascend", 1)
        >>> ms.runtime.launch_blocking()
    """
    return RuntimeConf.get_instance().set_launch_blocking()


@args_type_check(threads_num=int)
def dispatch_threads_num(threads_num):
    """
    Set the threads number of runtime used.

    The framework set the runtime number of threads are 5 by default.

    Args:
        threads_num (int): The threads number of runtime used.

    Examples:
        >>> import mindspore as ms
        >>> ms.set_device("Ascend", 1)
        >>> ms.runtime.dispatch_threads_num(6)
    """
    if RuntimeConf.get_instance().is_dispatch_threads_num_configured():
        raise RuntimeError("The 'dispatch_threads_num' can not be set repeatedly.")

    threads_num = Validator.check_positive_int(threads_num, "threads_num")

    return RuntimeConf.get_instance().set_dispatch_threads_num(threads_num)


def set_cpu_affinity(enable_affinity, affinity_cpu_list=None, module_to_cpu_dict=None):
    """
    Enable thread-level core binding to allocate specific CPU cores for key MindSpore modules (main thread, pynative,
    runtime, and minddata), preventing performance instability caused by CPU core contention among MindSpore threads.

    Note:
        - Flexible Core Binding Configuration:

          1. When `affinity_cpu_list` is not specified, the process automatically determines the CPU affinity range
             based on available CPU cores, NUMA nodes, and device resources in the environment.
          2. When `affinity_cpu_list` is specified, the process manually binds to the CPU range defined in
             `affinity_cpu_list`.
          3. When `module_to_cpu_dict` is not specified, the default bind-core strategy assigns the CPU
             cores to the `"main"` module.
          4. When `module_to_cpu_dict` is specified, the process manually binds each module to CPU ranges as
             defined in `module_to_cpu_dict`.
        - The automated bind-core strategy generation scenario invokes system commands to obtain CPU, NUMA node, and
          device resources on the environment, and some commands cannot be executed successfully due to environment
          differences; the automated bind-core strategy generated will vary according to the resources available on the
          environment:

          1. `cat /sys/fs/cgroup/cpuset/cpuset.cpus`, to obtain the available CPU resources on the environment; if the
             execution of this command fails, the bind-core function will not take effect.
          2. `npu-smi info -m`, get the available NPU resources on the environment; if the execution of this command
             fails, the bind-core strategy will be generated only based on the available CPU resources,
             without considering the device affinity.
          3. `npu-smi info -t board -i {NPU_ID} -c {CHIP_ID}`, get NPU details based on the logical ID of the device;
             if the execution of this command fails, the bind-core strategy is generated based on the available CPU
             resources only, regardless of device affinity.
          4. `lspci -s {PCIe_No} -vvv`, get the hardware information of the device on the environment; if the execution
             of this command fails, the bind-core strategy is generated only based on the available CPU resources,
             without considering the device affinity.
          5. `lscpu`, get information about CPUs and NUMA nodes on the environment; if the execution of this command
             fails, only the available CPU resources are used to generate the bind-core strategy, without considering
             the device affinity.

    Args:
        enable_affinity (bool): Enables/disables thread-level core binding.
        affinity_cpu_list (list, optional): Manually specifies the CPU affinity range for the process. Format:
            `["cpuidX-cpuidY"]` (e.g., ``["0-3", "8-11"]``). Default: ``None`` (uses auto-generated binding strategy
            based on system resources). Passing an empty list `[]` behaves the same as ``None``.
        module_to_cpu_dict (dict, optional): Customizes core binding for specific modules. Valid keys
            (module names) are ``"main"``, ``"runtime"``, ``"pynative"``, ``"minddata"``. Valid value is a list
            of ``int`` indices representing CPU cores (e.g., ``{"main": [0,1], "minddata": [6,7]}``).
            Default: ``None`` (automatically binds core for module `"main"`). Passing an empty dict `{}`
            behaves the same as ``None``.

    Raises:
        TypeError: The `enable_affinity` parameter is not a boolean.
        TypeError: The `affinity_cpu_list` parameter is neither a list nor ``None``.
        TypeError: An element in `affinity_cpu_list` is not a string.
        ValueError: An element in `affinity_cpu_list` does not follow the ``["cpuidX-cpuidY"]`` format.
        TypeError: The `module_to_cpu_dict` parameter is neither a dictionary nor ``None``.
        TypeError: A key in `module_to_cpu_dict` is not a string.
        TypeError: A value in `module_to_cpu_dict` is not a list.
        ValueError: An element in `module_to_cpu_dict` values is not a non-negative integer.
        RuntimeError: In custom core binding scenarios, the specified CPU cores for a device are unavailable
            in the environment.
        RuntimeError: The `mindspore.runtime.set_cpu_affinity` API is called repeatedly.

    Examples:
        >>> import mindspore as ms
        >>> ms.set_device("Ascend", 1)
        >>> ms.runtime.set_cpu_affinity(True)
        >>>
        >>> import mindspore as ms
        >>> ms.set_device("Ascend", 1)
        >>> ms.runtime.set_cpu_affinity(True, ["10-19", "23-40"])
        >>>
        >>> import mindspore as ms
        >>> ms.set_device("Ascend", 1)
        >>> ms.runtime.set_cpu_affinity(True, ["10-19", "23-40"], {"main": [0,1,2,3], "runtime": [4,5,6]})
    """
    affinity_cpu_list = _adapt_to_dict(affinity_cpu_list)
    if affinity_cpu_list is False:
        return

    _validate_affinity_cpu_list(affinity_cpu_list)
    _validate_module_cpu_index(module_to_cpu_dict)

    if RuntimeConf.get_instance().is_thread_bind_core_configured():
        raise RuntimeError("The 'mindspore.runtime.set_cpu_affinity' cannot be set repeatedly.")
    if not enable_affinity:
        RuntimeConf.get_instance().set_thread_bind_core_configured()
        return
    module_bind_core_strategy = _get_cpu_affinity_strategy(affinity_cpu_list, module_to_cpu_dict)
    if not module_bind_core_strategy:
        logger.warning("set_cpu_affinity is not enabled because the environment does not meet the "
                       "basic conditions for binding core.")
        RuntimeConf.get_instance().set_thread_bind_core_configured()
        return
    RuntimeConf.get_instance().thread_bind_core(module_bind_core_strategy)


@args_type_check(thread_num=int, kernel_group_num=int)
def set_kernel_launch_group(thread_num=2, kernel_group_num=8):
    """
    O0 mode supports operator batch parallel delivery interface, supports enabling
    parallel delivery, and configures parallel number.

    Args:
        thread_num (int, optional): The number of concurrent threads, generally not recommended
            to increase. The `thread_num` and the number of threads configured by the existing interface
            mindspore.runtime.dispatch_threads_num are independent of each other. Default value is ``2``.
        kernel_group_num (int, optional): Total number of operator groups,
            kernel_group_num/thread_num groups per thread. Default value is ``8``.

    Examples:
        >>> import mindspore as ms
        >>> ms.runtime.set_kernel_launch_group(thread_num=2, kernel_group_num=8)
    """
    if RuntimeConf.get_instance().is_kernel_launch_group_configured():
        raise RuntimeError("The 'kernel_launch_group' can not be set repeatedly.")

    if RuntimeConf.get_instance().get_enable_kernel_launch_capture():
        raise RuntimeError("The kernel launch group and kernel launch capture can not be set together")

    if thread_num < 1:
        raise ValueError(f"The value of thread_num should be at least 1, but got {thread_num}")

    if kernel_group_num < 1:
        raise ValueError(f"The value of kernel_group_num should be at least 1, but got {kernel_group_num}")

    if (kernel_group_num % thread_num) != 0:
        raise ValueError(f"Invalid parameter value, kernel_group_num: {kernel_group_num} cannot "
                         f"be evenly divisible by thread_num: {thread_num}")

    return RuntimeConf.get_instance().set_kernel_launch_group(thread_num, kernel_group_num)


@args_type_check(enable_capture_graph=bool)
def set_kernel_launch_capture(enable_capture_graph, op_capture_skip=None):
    """
    In O0/O1 mode, the incremental inference scenario supports graph capture.
    By capturing the CPU-side operator dispatch behavior into a graph,
    the performance of CPU-side operator dispatch is improved.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        enable_capture_graph (bool): Whether to enable graph capture.
            It can be turned on or off at any position in the script.
        op_capture_skip (list): Custom non-captured operator names. Default: ``None``.

    Examples:
        >>> import mindspore as ms
        >>> op_capture_skip = ['matmul', 'addn']
        >>> ms.runtime.set_kernel_launch_capture(True, op_capture_skip)
    """
    if RuntimeConf.get_instance().is_kernel_launch_group_configured():
        raise RuntimeError("The kernel launch group and kernel launch capture can not be set together")

    if op_capture_skip is None:
        op_capture_skip = []

    if not isinstance(op_capture_skip, list):
        raise TypeError("op_capture_skip must be a list")

    return RuntimeConf.get_instance().set_kernel_launch_capture(enable_capture_graph, op_capture_skip)
