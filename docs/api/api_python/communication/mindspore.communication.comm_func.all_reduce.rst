mindspore.communication.comm_func.all_reduce
============================================

.. py:function:: mindspore.communication.comm_func.all_reduce(tensor, op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP)

    使用指定方式对通信组内的所有设备的Tensor数据进行规约操作，所有设备都得到相同的结果，返回规约操作后的张量。

    .. note::
        集合中的所有进程的Tensor必须具有相同的shape和格式。

    参数：
        - **tensor** (Tensor) - 输入待规约操作的Tensor，Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **op** (str，可选) - 规约的具体操作。如 ``"sum"`` 、 ``"prod"`` 、 ``"max"`` 、和 ``"min"`` 。默认值： ``ReduceOp.SUM`` 。
        - **group** (str，可选) - 工作的通信组。默认值：``GlobalComm.WORLD_COMM_GROUP`` （即Ascend平台为 ``"hccl_world_group"`` ，GPU平台为 ``"nccl_world_group"`` ）。

    返回：
        Tensor，shape与输入相同，即 :math:`(x_1, x_2, ..., x_R)` 。其内容取决于操作。

    异常：
        - **TypeError** - 首个输入的数据类型不为Tensor，`op` 或 `group` 不是str。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

        该样例需要在2卡环境下运行。
