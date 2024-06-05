mindspore.ops.Barrier
=====================

.. py:class:: mindspore.ops.Barrier(group=GlobalComm.WORLD_COMM_GROUP)

    同步通信域内的多个进程。进程调用到该算子后进入阻塞状态，直到通信域内所有进程调用到该算子，
    进程被唤醒并继续执行。

    参数：
        - **group** (str，可选) - 表示通信域。默认值： ``GlobalComm.WORLD_COMM_GROUP`` 。

    异常：
        - **TypeError** - 首个输入的数据类型不为Tensor，`op` 或 `group` 不是str。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。
