mindspore_lite.Context
======================

.. py:class:: mindspore_lite.Context(thread_num=None, inter_op_parallel_num=None, thread_affinity_mode=None, thread_affinity_core_list=None, enable_parallel=False)

    Context用于在执行期间传递环境变量。

    在运行程序之前，应配置context。如果没有配置，默认情况下将根据设备目标进行自动设置。

    .. note::
        如果同时设置 `thread_affinity_core_list` 和 `thread_affinity_mode` 在同一个context中，则 `thread_affinity_core_list` 生效，
        但 `thread_affinity_mode` 无效。

    参数：
        - **thread_num** (int，可选) - 设置运行时的线程数。 `thread_num` 不能小于 `inter_op_parallel_num` 。将 `thread_num` 设置为0表示 `thread_num` 将基于计算机性能和核心数自动调整。默认值：None，等同于设置为0。
        - **inter_op_parallel_num** (int，可选) - 设置运行时算子的并行数。 `inter_op_parallel_num` 不能大于 `thread_num` 。将 `inter_op_parallel_num` 设置为0表示 `inter_op_parallel_num` 将基于计算机性能和核心数自动调整。默认值：None，等同于设置为0。
        - **thread_affinity_mode** (int，可选) - 设置运行时的CPU/GPU/NPU绑核策略模式。支持以下 `thread_affinity_mode` 。默认值：None，等同于设置为0。

          - **0** - 不绑核。
          - **1** - 绑大核优先。
          - **2** - 绑中核优先。

        - **thread_affinity_core_list** (list[int]，可选) - 设置运行时的CPU/GPU/NPU绑核策略列表。例如：[0,1]在CPU设备上代表指定绑定0号CPU和1号CPU。默认值：None，等同于设置为[]。
        - **enable_parallel** (bool，可选) - 设置状态是否启用并行执行模型推理或并行训练。默认值：False。

    异常：
        - **TypeError** - `thread_num` 既不是int类型也不是None。
        - **TypeError** - `inter_op_parallel_num` 既不是int类型也不是None。
        - **TypeError** - `thread_affinity_mode` 既不是int类型也不是None。
        - **TypeError** - `thread_affinity_core_list` 既不是list类型也不是None。
        - **TypeError** - `thread_affinity_core_list` 是list类型，但元素既不是int类型也不是None。
        - **TypeError** - `enable_parallel` 不是bool类型。
        - **ValueError** - `thread_num` 小于0。
        - **ValueError** - `inter_op_parallel_num` 小于0。

    .. py:method:: append_device_info(device_info)

        将一个用户定义的设备信息附加到上下文中。

        .. note::
            添加GPU设备信息后，必须在调用上下文之前添加CPU设备信息。因为当GPU不支持算子时，系统将尝试CPU是否支持该算子。此时，需要切换至带有CPU设备信息的上下文中。

            添加Ascend设备信息后，在原始模型的输入format与转换生成的模型的输入format不一致的场景时，用户可以选择在调用上下文之前添加CPU设备硬件信息。因为在这种情况下，在Ascend设备上转换生成的模型中将包含 `Transpose` 节点，该节点目前需要在CPU上执行推理，因此需要切换至带有CPU设备信息的上下文中。

        参数：
            - **device_info** (DeviceInfo) - 实例化的设备信息。

        异常：
            - **TypeError** - `device_info` 不是DeviceInfo类型。
