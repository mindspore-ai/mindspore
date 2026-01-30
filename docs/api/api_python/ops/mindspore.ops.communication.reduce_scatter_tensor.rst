mindspore.ops.communication.reduce_scatter_tensor
==================================================

.. py:function:: mindspore.ops.communication.reduce_scatter_tensor(output, input, op=ReduceOp.SUM, group=None, async_op=False)

    对指定通信组中的张量进行归约和分散操作，并返回归约和分散后的张量。

    .. note::
        - 集合中所有进程的Tensor必须具有相同的shape和格式。

    参数：
        - **output** (Tensor) - 输出张量，与 `input` 具有相同的数据类型，shape为 :math:`(N/rank\_size, *)`。如果函数以非in-place模式运行，此参数无效。
        - **input** (Tensor) - 待归约和分散的输入张量，假设其shape为 :math:`(N, *)`，其中 `*` 表示任意数量的额外维度。N必须能被rank_size整除。rank_size指通信组中的卡数。
        - **op** (str，可选) - 用于逐元素归约的操作，如SUM和MAX。默认值： ``ReduceOp.SUM`` 。
        - **group** (str，可选) - 通信组名称，如果为 ``None`` ，Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        - CommHandle，若函数以in-place模式运行，返回它。若 `async_op` 是True，CommHandle是一个异步工作句柄。若 `async_op` 是False，CommHandle将返回None。
        - Tuple(Tensor, CommHandle)，若函数以非in-place模式运行，返回它。输出张量与 `input` 具有相同的数据类型，shape为 :math:`(N/rank\_size, *)`。若 `async_op` 是True，CommHandle是一个异步工作句柄。若 `async_op` 是False，CommHandle将返回None。

    异常：
        - **TypeError** - 输入和输出参数的类型不是Tensor，`op` 或 `group` 不是str，async_op不是bool或'op'无效。
        - **ValueError** - 如果输入的第一维不能被rank_size整除。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
