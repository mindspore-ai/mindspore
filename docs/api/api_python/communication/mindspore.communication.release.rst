mindspore.communication.release
===============================

.. py:function:: mindspore.communication.release()

    释放分布式资源，例如 `HCCL`、`NCCL` 或 `MCCL` 服务。

    .. note::
        `release` 方法应该在 `init()` 方法之后使用。如果不使用此方法，则在程序结束时资源才会自动释放。

    异常：
        - **RuntimeError** - 在释放分布式资源失败时抛出。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst
