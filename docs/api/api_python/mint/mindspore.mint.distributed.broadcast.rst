mindspore.mint.distributed.broadcast
=====================================

.. py:function:: mindspore.mint.distributed.broadcast(tensor, src, group=None, async_op=False)

    对输入数据整组广播。

    .. note::
        - 集合中的所有进程的Tensor的shape和数据格式必须相同。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **tensor** (Tensor) - 待广播出去的或接收广播的Tensor。
        - **src** (int) - 表示发送源的进程编号。只有该进程会广播张量。
        - **group** (str，可选) - 通信组名称，如果为 ``None`` ，Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        CommHandle，若 `async_op` 是True，CommHandle是一个异步工作句柄。若 `async_op` 是False，CommHandle将返回None。

    异常：
        - **TypeError** - `tensor` 的数据类型不为Tensor，`src` 不是int， `group` 不是str或  `async_op` 不是bool。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
