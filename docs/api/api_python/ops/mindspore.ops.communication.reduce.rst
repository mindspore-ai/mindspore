mindspore.ops.communication.reduce
===================================

.. py:function:: mindspore.ops.communication.reduce(tensor, dst, op=ReduceOp.SUM, group=None, async_op=False)

    对指定通信组中的进程进行张量归约操作，将结果发送到目标dst（全局rank），并返回发送到目标进程的张量。

    .. note::
        - 只有目标rank的进程会接收归约后的输出。
        - 当前仅支持PyNative模式，不支持Graph模式。
        - 其他进程只会得到一个shape为[1]的张量，没有数学意义。

    参数：
        - **tensor** (Tensor) - 集合操作的输入和输出。该函数以in-place模式运行。
        - **dst** (int) - 接收归约输出的进程的目标rank（全局rank）。
        - **op** (str，可选) - 用于逐元素归约的操作，如 ``"sum"`` 、 ``"prod"`` 、 ``"max"`` 、 ``"min"`` 和 ``"avg"`` 。使用 ``"avg"`` 时，结果为求和后除以组内设备数。默认值： ``ReduceOp.SUM`` 。
        - **group** (str，可选) - 通信组名称。默认值： ``None`` ，即Ascend平台表示为 ``"hccl_world_group"`` 。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        CommHandle。若 `async_op` 是 ``True`` ，CommHandle是一个异步工作句柄。若 `async_op` 是 ``False`` ，CommHandle将返回 ``None`` 。

    异常：
        - **TypeError** - `tensor` 的类型不是Tensor，`op` 或 `group` 不是str， `async_op` 不是bool，或 `op` 无效。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: ../mint/mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
