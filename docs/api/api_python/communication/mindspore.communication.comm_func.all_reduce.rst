mindspore.communication.comm_func.all_reduce
============================================

.. py:function:: mindspore.communication.comm_func.all_reduce(tensor, op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP, async_op=False)

    使用指定方式对通信组内的所有设备的Tensor数据进行归约操作，所有设备都得到相同的结果，返回归约操作后的Tensor。

    .. note::
        集合中的所有进程的Tensor必须具有相同的shape和格式。

    参数：
        - **tensor** (Tensor) - 输入待归约操作的Tensor，shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **op** (str, 可选) - 归约的具体操作。如 ``"sum"`` 、 ``"prod"`` 、 ``"max"`` 和 ``"min"`` 。默认值： ``ReduceOp.SUM`` 。
        - **group** (str, 可选) - 工作的通信组。默认值：``GlobalComm.WORLD_COMM_GROUP`` （即Ascend平台为 ``"hccl_world_group"`` ，GPU平台为 ``"nccl_world_group"`` ）。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        Tuple(Tensor, CommHandle)，输出shape与输入相同，即 :math:`(x_1, x_2, ..., x_R)` 。其内容取决于操作。
        若 `async_op` 是True，CommHandle是一个异步工作句柄；若 `async_op` 是False，CommHandle将返回None。

    异常：
        - **TypeError** - 首个输入的数据类型不为Tensor， 或 `op` 与 `group` 中任一不是字符串。
        - **RuntimeError** - 目标设备无效、后端无效，或分布式初始化失败。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

        该样例需要在2卡环境下运行。
