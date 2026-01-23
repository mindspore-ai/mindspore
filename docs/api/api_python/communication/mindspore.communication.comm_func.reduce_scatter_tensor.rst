mindspore.communication.comm_func.reduce_scatter_tensor
=======================================================

.. py:function:: mindspore.communication.comm_func.reduce_scatter_tensor(tensor, op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP, async_op=False)

    归约并且分发指定通信组中的Tensor，返回分发后的Tensor。

    .. note::
        集合中所有进程的Tensor必须具有相同的shape和格式。

    参数：
        - **tensor** (Tensor) - 输入待归约且分发的Tensor，假设其形状为 :math:`(N, *)` ，其中 `*` 为任意数量的额外维度。N必须能够被rank_size整除，rank_size为当前通讯组里面的计算卡数量。
        - **op** (str, 可选) - 归约的具体操作。如 ``"sum"`` 、 ``"prod"`` 、 ``"max"`` 和 ``"min"`` 。默认值： ``ReduceOp.SUM`` 。
        - **group** (str, 可选) - 工作的通信组。默认值： ``GlobalComm.WORLD_COMM_GROUP`` （即Ascend平台为 ``"hccl_world_group"`` ，GPU平台为 ``"nccl_world_group"`` ）。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        Tuple(Tensor, CommHandle)，输出Tensor数据类型与 `tensor` 一致，shape为 :math:`(N/rank\_size, *)` 。
        若 `async_op` 是True，CommHandle是一个异步工作句柄；若 `async_op` 是False，CommHandle将返回None。

    异常：
        - **TypeError** - 首个输入的数据类型不为Tensor，或 `op` 与 `group` 中任一不是字符串。
        - **ValueError** - 输入的第一个维度不能被rank size整除。
        - **RuntimeError** - 目标设备无效、后端无效或分布式初始化失败。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

        该样例需要在2卡环境下运行。
