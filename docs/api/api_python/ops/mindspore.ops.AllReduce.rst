mindspore.ops.AllReduce
========================

.. py:class:: mindspore.ops.AllReduce(op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP)

    使用指定方式对通信组内的所有设备的Tensor数据进行规约操作，所有设备都得到相同的结果。

    .. note::
        集合中的所有进程的Tensor必须具有相同的shape和格式。

    参数：
        - **op** (str) - 规约的具体操作，如"sum"、"max"、和"min"。默认值：ReduceOp.SUM。
        - **group** (str) - 工作的通信组。默认值："GlobalComm.WORLD_COMM_GROUP"（即Ascend平台为"hccl_world_group"，GPU平台为"nccl_world_group" ）。

    输入：
        - **input_x** (Tensor) - shape为 :math:`(x_1, x_2, ..., x_R)` 的Tensor。

    输出：
        Tensor，shape与输入相同，即 :math:`(x_1, x_2, ..., x_R)` 。其内容取决于操作。

    异常：
        - **TypeError** - `op` 或 `group` 不是str，或者输入的数据类型是bool。

    样例：

    .. note::
        .. include:: mindspore.ops.comm_note.rst

        该样例需要在多卡环境下运行。