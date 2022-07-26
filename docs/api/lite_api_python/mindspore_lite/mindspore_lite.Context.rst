mindspore_lite.Context
======================

.. py:class:: mindspore_lite.Context(thread_num=None, inter_op_parallel_num=None, thread_affinity_mode=None, thread_affinity_core_list=None, enable_parallel=False)

    Context用于在执行期间存储环境变量。

    在运行程序之前，应配置context。如果没有配置，默认情况下将根据设备目标进行自动设置。

    .. note::
        如果同时设置thread_affinity_mode和thread_affinity_core_list，则thread_affinity_core_list有效，但thread_affinity_mode无效。
        参数默认值是None时表示不设置。

    参数：
        - **thread_num** (int，可选) - 设置运行时的线程数。默认值：None。
        - **inter_op_parallel_num** (int，可选) - 设置运行时算子的并行数。默认值：None。
        - **thread_affinity_mode** (int，可选) - 与CPU核心的线程亲和模式。默认值：None。

          - **0** - 无亲和性。
          - **1** - 大核优先。
          - **2** - 小核优先。

        - **thread_affinity_core_list** (list[int]，可选) - 与CPU核心的线程亲和列表。默认值：None。
        - **enable_parallel** (bool，可选) - 设置状态是否启用并行执行模型推理或并行训练。默认值：False。

    异常：
        - **TypeError** - `thread_num` 不是int类型或None。
        - **TypeError** - `inter_op_parallel_num` 不是int类型或None。
        - **TypeError** - `thread_affinity_mode` 不是int类型或None。
        - **TypeError** - `thread_affinity_core_list` 不是list类型或None。
        - **TypeError** - `thread_affinity_core_list` 是list类型，但元素不是int类型或None。
        - **TypeError** - `enable_parallel` 不是bool类型。
        - **ValueError** - `thread_num` 小于0。
        - **ValueError** - `inter_op_parallel_num` 小于0。

    .. py:method:: append_device_info(device_info)

        设置张量的数据类型。

        参数：
            - **device_info** (DeviceInfo) - 实例化的设备信息。

        异常：
            - **TypeError** - `device_info` 不是DeviceInfo类型。
