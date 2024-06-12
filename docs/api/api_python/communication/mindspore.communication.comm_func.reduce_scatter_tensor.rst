mindspore.communication.comm_func.reduce_scatter_tensor
=======================================================

.. py:function:: mindspore.communication.comm_func.reduce_scatter_tensor(tensor, op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP)

    规约并且分发指定通信组中的张量，返回分发后的张量。

    .. note::
        在集合的所有过程中，Tensor必须具有相同的shape和格式。

    参数：
        - **tensor** (Tensor) - 输入待规约且分发的Tensor，假设其形状为 :math:`(N, *)` ，其中 `*` 为任意数量的额外维度。N必须能够被rank_size整除，rank_size为当前通讯组里面的计算卡数量。
        - **op** (str, 可选) - 规约的具体操作。如 ``"sum"`` 、 ``"prod"`` 、 ``"max"`` 、和 ``"min"`` 。默认值： ``ReduceOp.SUM`` 。
        - **group** (str，可选) - 工作的通信组，默认值： ``GlobalComm.WORLD_COMM_GROUP`` （即Ascend平台为 ``"hccl_world_group"`` ，GPU平台为 ``"nccl_world_group"`` ）。

    返回：
        Tensor，数据类型与 `input_x` 一致，shape为 :math:`(N/rank\_size, *)` 。

    异常：
        - **TypeError** - 首个输入的数据类型不为Tensor，`op` 和 `group` 不是字符串。
        - **ValueError** - 如果输入的第一个维度不能被rank size整除。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

        该样例需要在2卡环境下运行。
