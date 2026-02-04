mindspore.mint.distributed.reduce_scatter_tensor
==================================================

.. py:function:: mindspore.mint.distributed.reduce_scatter_tensor(output, input, op=ReduceOp.SUM, group=None, async_op=False)

    归约并且分发指定通信组中的Tensor，返回分发后的Tensor。

    .. note::
        - 在集合的所有过程中，Tensor必须具有相同的shape和格式。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **output** (Tensor) - 输出分发的Tensor，其shape为 :math:`(N/rank\_size, *)`。
        - **input** (Tensor) - 输入待归约且分发的Tensor，假设其shape为 :math:`(N, *)` ，其中 `*` 为任意数量的额外维度。N必须能够被rank_size整除，rank_size为当前通讯组里面的计算卡数量。
        - **op** (str, 可选) - 归约的具体操作。如 ``"sum"`` 、 ``"prod"`` 、 ``"max"`` 、和 ``"min"`` 。默认值： ``ReduceOp.SUM`` 。
        - **group** (str，可选) - 通信组名称，如果为 ``None`` ，Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        CommHandle，若 `async_op` 是True，CommHandle是一个异步工作句柄。若 `async_op` 是False，CommHandle将返回None。

    异常：
        - **TypeError** - 首个输入的数据类型不为Tensor，`op` 和 `group` 不是字符串， `async_op` 不是bool， `op` 值非法。
        - **ValueError** - 如果输入的第一个维度不能被rank size整除。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
