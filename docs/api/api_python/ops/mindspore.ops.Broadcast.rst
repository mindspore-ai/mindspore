mindspore.ops.Broadcast
========================

.. py:class:: mindspore.ops.Broadcast(root_rank, group=GlobalComm.WORLD_COMM_GROUP)

    对输入数据整组广播。

    .. note::
        集合中的所有进程的Tensor的shape和数据格式相同。在运行下面样例时，用户需要预设通信环境变量，请在 `MindSpore \
        <https://www.mindspore.cn/docs/zh-CN/r1.9/api_python/mindspore.ops.html#通信算子>`_ 官网上查看详情。

    参数：
        - **root_rank** (int) - 表示发送源的进程编号。除发送数据的进程外，存在于所有进程中。
        - **group** (str) - 表示通信域。默认值："hccl_world_group"。

    输入：
        - **input_x** (Tensor) - Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。

    输出：
        Tensor，shape与输入相同，即 :math:`(x_1, x_2, ..., x_R)` 。内容取决于 `root_rank` device的数据。

    异常：
        - **TypeError** - root_rank不是int或group不是str。