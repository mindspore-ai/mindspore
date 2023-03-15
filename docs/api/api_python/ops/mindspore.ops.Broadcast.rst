mindspore.ops.Broadcast
========================

.. py:class:: mindspore.ops.Broadcast(root_rank, group=GlobalComm.WORLD_COMM_GROUP)

    对输入数据整组广播。

    .. note::
        集合中的所有进程的Tensor的shape和数据格式相同。

    参数：
        - **root_rank** (int) - 表示发送源的进程编号。除发送数据的进程外，存在于所有进程中。
        - **group** (str) - 表示通信域。默认值："hccl_world_group"。

    输入：
        - **input_x** (tuple[Tensor]) - Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。

    输出：
        tuple[Tensor]，Tensor的shape与输入相同，即 :math:`(x_1, x_2, ..., x_R)` 。内容取决于 `root_rank` device的数据。

    异常：
        - **TypeError** - root_rank不是int或group不是str。

    样例：

    .. note::
        .. include:: mindspore.ops.comm_note.rst

        该样例需要在多卡环境下运行。