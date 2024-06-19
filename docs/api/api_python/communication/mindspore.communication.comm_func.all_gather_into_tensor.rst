mindspore.communication.comm_func.all_gather_into_tensor
========================================================

.. py:function:: mindspore.communication.comm_func.all_gather_into_tensor(tensor, group=GlobalComm.WORLD_COMM_GROUP)

    汇聚指定的通信组中的Tensor，并返回汇聚后的张量。

    .. note::
        - 集合中所有进程的Tensor必须具有相同的shape和格式。

    参数：
        - **tensor** (Tensor) - 输入待汇聚操作的Tensor，Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **group** (str，可选) - 工作的通信组，默认值： ``GlobalComm.WORLD_COMM_GROUP`` （即Ascend平台为 ``"hccl_world_group"`` ，GPU平台为 ``"nccl_world_group"`` ）。

    返回：
        Tensor，如果组中的device数量为N，则输出的shape为 :math:`(N, x_1, x_2, ..., x_R)` 。

    异常：
        - **TypeError** - 首个输入的数据类型不为Tensor，`group` 不是str。
        - **ValueError** - 调用进程的rank id大于本通信组的rank大小。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

        该样例需要在2卡环境下运行。
