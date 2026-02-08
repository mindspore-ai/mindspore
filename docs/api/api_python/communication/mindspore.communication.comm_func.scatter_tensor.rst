mindspore.communication.comm_func.scatter_tensor
================================================

.. py:function:: mindspore.communication.comm_func.scatter_tensor(tensor, src=0, group=GlobalComm.WORLD_COMM_GROUP)

    将输入的Tensor均匀散射到通信域的卡上。

    .. note::
        - 该接口只支持Tensor输入，且只支持均匀切分。
        - 只有源为 `src` 的进程（全局的进程编号）才会将输入Tensor作为散射源。
        - 当前支持PyNative模式，不支持Graph模式。

    参数：
        - **tensor** (Tensor) - 输入待散射的Tensor，shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **src** (int, 可选) - 表示发送源的进程编号。只有该进程会发送散射源Tensor。默认值：``0``。
        - **group** (str, 可选) - 工作的通信组。默认值：``GlobalComm.WORLD_COMM_GROUP`` （即Ascend平台为 ``"hccl_world_group"`` ，GPU平台为 ``"nccl_world_group"`` ）。

    返回：
        Tensor，其shape为 :math:`(x_1/src\_rank, x_2, ..., x_R)` 。
        Tensor第0维等于输入数据第0维除以通信组内的进程数量，其他维度相同。

    异常：
        - **TypeError** - 首个输入的数据类型不为Tensor，或 `group` 不是str。
        - **RuntimeError** - 目标设备无效、后端无效，或分布式初始化失败。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

        该样例需要在多卡环境下运行。
