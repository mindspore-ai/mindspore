mindspore.mint.distributed.barrier
=====================================

.. py:function:: mindspore.mint.distributed.barrier(group=None, async_op=False, device_ids=None)

    同步通信域内的多个进程。进程调用到该算子后进入阻塞状态，直到通信域内所有进程调用到该算子，
    进程被唤醒并继续执行。

    .. note::
        当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **group** (str，可选) - 通信组名称，如果为 ``None`` ， Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。
        - **device_ids** (list[int], 可选) - 当前为预留参数。默认值： ``None`` 。

    返回：
        CommHandle，若 `async_op` 是True，CommHandle是一个异步工作句柄。若 `async_op` 是False，CommHandle将返回None。

    异常：
        - **TypeError** - `group` 不是str， `async_op` 不是bool。
        - **RuntimeError** - 如果后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
