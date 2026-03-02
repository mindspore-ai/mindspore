mindspore.mint.distributed.reduce_scatter
============================================

.. py:function:: mindspore.mint.distributed.reduce_scatter(output, input_list, op=ReduceOp.SUM, group=None, async_op=False)

    归约并且分发指定通信组中的tensor，返回分发后的tensor。

    .. note::
        当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **output** (Tensor) - 输出归约且分发的tensor。
        - **input_list** (list[Tensor]) - 输入待归约且分发的tensor列表。
        - **op** (str, 可选) - 归约的具体操作。如 ``"sum"`` 、 ``"prod"`` 、 ``"max"`` 和 ``"min"`` 。默认值： ``ReduceOp.SUM`` 。
        - **group** (str，可选) - 通信组名称。默认值： ``None`` ，即在Ascend平台表示为 ``"hccl_world_group"`` 。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        CommHandle。若 `async_op` 是 ``True``，CommHandle是一个异步工作句柄；若 `async_op` 是 ``False``，CommHandle将返回 ``None``。

    异常：
        - **TypeError** - 输入 `output` 的数据类型不为Tensor， `input_list` 不为Tensor列表。
        - **TypeError** - `op` 和 `group` 不是字符串， `async_op` 不是bool， `op` 值非法。
        - **TypeError** - `input_list` 的大小不为通信组大小。
        - **TypeError** -  `output` 的数据类型和shape与 `input_list` 中所有元素存在不一致。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
