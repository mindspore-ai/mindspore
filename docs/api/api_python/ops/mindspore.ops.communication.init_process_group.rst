mindspore.ops.communication.init_process_group
================================================

.. py:function:: mindspore.ops.communication.init_process_group(backend="hccl", init_method=None, timeout=None, world_size=-1, rank=-1, store=None, group_name="", pg_options=None, device_id=None)

    初始化集合通信库，并创建一个默认的集合通信组。

    .. note::
        - 此方法在GPU和CPU版本的MindSpore中不受支持。
        - 在Ascend硬件平台上，此API应在定义任何Tensor和Parameter之前设置，以及在实例化和执行任何操作和网络之前设置。
        - MCCL通信组不支持重复初始化和销毁。

    参数：
        - **backend** (str, 可选) - 要使用的后端。默认值为 ``"hccl"``，目前仅支持 ``"hccl"``。
        - **init_method** (str, 可选) - 指定如何初始化集合通信组的URL。默认值： ``None``。
        - **timeout** (timedelta, 可选) - API执行的超时时间。默认值： ``None``。目前，此参数仅在主机端集群网络配置中使用 `init_method` 或 `store` 时支持。
        - **world_size** (int, 可选) - 参与作业的进程数。默认值： ``-1``。
        - **rank** (int, 可选) - 当前进程的rank。默认值： ``-1``。
        - **store** (Store, 可选) - 存储键/值数据的对象，便于交换进程间通信地址和连接信息。默认值： ``None``。目前，仅支持 ``TCPStore`` 类型。
        - **group_name** (str, 可选) - 设置默认全局通信域名称。默认值： ``""``。
        - **pg_options** (ProcessGroupOptions, 无效, 可选) - 进程组选项，指定在构建特定进程组期间需要传递的附加选项。提供的参数是预留参数，当前设置不生效。默认值： ``None``。
        - **device_id** (int, 无效, 可选) - 要执行的设备id。提供的参数是预留参数，当前设置不生效。默认值： ``None``。

    异常：
        - **ValueError** - 如果 `backend` 不是 ``"hccl"``。
        - **ValueError** - 如果 `world_size` 不等于-1或进程组数。
        - **ValueError** - 如果同时设置了 `init_method` 和 `store`。
        - **ValueError** - 当使用初始化方法 `init_method` 或 `store` 时，`world_size` 未正确设置为正整数。
        - **ValueError** - 当使用初始化方法 `init_method` 或 `store` 时，`rank` 未正确设置为非负整数。
        - **RuntimeError** - 如果设备目标无效，或后端无效，或分布式初始化失败，或当后端为HCCL时未导出环境变量RANK_ID/MINDSPORE_HCCL_CONFIG_PATH。

    样例：

    .. note::
        .. include:: ../mint/mindspore.mint.comm_note.rst
