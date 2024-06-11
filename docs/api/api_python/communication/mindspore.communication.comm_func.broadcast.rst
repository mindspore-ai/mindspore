mindspore.communication.comm_func.broadcast
===========================================

.. py:function:: mindspore.communication.comm_func.broadcast(tensor, src=0, group=GlobalComm.WORLD_COMM_GROUP)

    对输入数据整组广播。

    .. note::
        集合中的所有进程的Tensor的shape和数据格式必须相同。
        当前支持PyNative模式，不支持Graph模式。

    参数：
        - **tensor** (Tensor) - 输入待广播的Tensor。Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **src** (int，可选) - 表示发送源的进程编号。只有该进程会广播张量。默认值：0。
        - **group** (str，可选) - 表示通信域。默认值： ``GlobalComm.WORLD_COMM_GROUP`` 。

    返回：
        Tensor，Tensor的shape与输入相同，即 :math:`(x_1, x_2, ..., x_R)` 。

    异常：
        - **TypeError** - src不是int或group不是str。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

        该样例需要在多卡环境下运行。
