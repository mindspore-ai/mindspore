mindspore.ops.CollectiveScatter
===============================

.. py:class:: mindspore.ops.CollectiveScatter(src_rank=0, group=GlobalComm.WORLD_COMM_GROUP)

    对输入数据的数据进行均匀散射到通信域的卡上。

    .. note::
        该接口和 `pytoch.distributed.scatter` 存在行为差异。该接口只支持Tensor输入，且只支持均匀切分。
        只有源为src_rank的进程(全局的进程编号)才会将输入张量作为散射源。

    参数：
        - **src_rank** (int，可选) - 表示发送源的进程编号。只有该进程会发送散射源Tensor。
        - **group** (str，可选) - 表示通信域。默认值： ``GlobalComm.WORLD_COMM_GROUP`` 。

    输入：
        - **input_x** (Tensor) - 输入待散射的Tensor。Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。

    输出：
        Tensor，Tensor第0维等于散射源输入第0维除以 `src_rank` ，其他shape维度相同
        即 :math:`(x_1/src_rank, x_2, ..., x_R)` 。

    异常：
        - **TypeError** - 首个输入的数据类型不为Tensor， `op` 或 `group` 不是str。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。
