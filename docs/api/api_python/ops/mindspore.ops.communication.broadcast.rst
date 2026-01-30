mindspore.ops.communication.broadcast
======================================

.. py:function:: mindspore.ops.communication.broadcast(tensor, src, group=None, async_op=False)

    将张量广播到整个通信组。

    .. note::
        - 集合中所有进程的Tensor必须具有相同的shape和格式。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **tensor** (Tensor) - 如果src是当前进程的rank，则为要发送的数据；否则为用于保存接收数据的张量。
        - **src** (int) - 指定广播张量的进程的rank（全局rank）。只有进程 `src` 会广播张量。
        - **group** (str，可选) - 通信组名称，如果为 ``None`` ，Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        CommHandle，若 `async_op` 是True，CommHandle是一个异步工作句柄。若 `async_op` 是False，CommHandle将返回None。

    异常：
        - **TypeError** - `tensor` 参数的类型不是Tensor，`src` 不是整数，`group` 不是字符串或 `async_op` 不是bool。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
