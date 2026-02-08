mindspore.communication.comm_func.barrier
=========================================

.. py:function:: mindspore.communication.comm_func.barrier(group=GlobalComm.WORLD_COMM_GROUP)

    同步通信域内的多个进程。进程调用到该算子后进入阻塞状态，直到通信域内所有进程调用到该算子，进程被唤醒并继续执行。

    参数：
        - **group** (str, 可选) - 工作的通信组。默认值： ``GlobalComm.WORLD_COMM_GROUP`` （即Ascend平台为 ``"hccl_world_group"`` ，GPU平台为 ``"nccl_world_group"`` ）。

    异常：
        - **RuntimeError** - 后端无效，或分布式初始化失败。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

        该样例需要在多卡环境下运行。
