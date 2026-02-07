mindspore.ops.communication.reduce_scatter
============================================

.. py:function:: mindspore.ops.communication.reduce_scatter(output, input_list, op=ReduceOp.SUM, group=None, async_op=False)

    对指定通信组中的张量进行归约和分散操作，并返回归约和分散后的张量。

    参数：
        - **output** (Tensor) - 输出张量。
        - **input_list** (list[Tensor]) - 要归约和分散的张量列表。
        - **op** (str，可选) - 用于逐元素归约的操作，如SUM和MAX。默认值： ``ReduceOp.SUM`` 。
        - **group** (str，可选) - 通信组名称。默认值： ``None`` ，即Ascend平台表示为 ``"hccl_world_group"`` 。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        CommHandle。若 `async_op` 是 ``True`` ，CommHandle是一个异步工作句柄。若 `async_op` 是 ``False`` ，CommHandle将返回 ``None`` 。

    异常：
        - **TypeError** - 如果 `output` 参数的类型不是Tensor，`input_list` 不是Tensor列表。
        - **TypeError** - 如果 `op` 和 `group` 不是str，`async_op` 不是bool或 `op` 无效。
        - **TypeError** - 如果 `input_list` 的大小不等于通信组大小。
        - **TypeError** - 如果 `output` 的类型或shape不等于 `input_list` 的成员。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: ../mint/mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
