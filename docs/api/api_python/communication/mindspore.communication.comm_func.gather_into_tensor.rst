mindspore.communication.comm_func.gather_into_tensor
====================================================

.. py:function:: mindspore.communication.comm_func.gather_into_tensor(tensor, dst=0, group=GlobalComm.WORLD_COMM_GROUP)

    对通信组的输入Tensor进行聚合。操作会将每张卡的输入Tensor在第0维度上进行聚合，发送到对应卡上。

    .. note::
        - 只有目标为 `dst` 的进程（全局的进程编号）才会收到聚合操作后的输出。其他进程只得到一个形状为[1]的Tensor，且该Tensor没有数学意义。
        - 当前支持PyNative模式，不支持Graph模式。

    参数：
        - **tensor** (Tensor) - 输入待聚合的Tensor，shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **dst** (int, 可选) - 表示发送目标的进程编号。只有该进程会接收聚合后的Tensor。默认值： ``0``。
        - **group** (str, 可选) - 工作的通信组。默认值： ``GlobalComm.WORLD_COMM_GROUP`` （即Ascend平台为 ``"hccl_world_group"`` ，GPU平台为 ``"nccl_world_group"`` ）。

    返回：
        Tensor，其shape为 :math:`(\sum x_1, x_2, ..., x_R)`。Tensor第0维等于输入数据第0维求和，其他shape相同。

    异常：
        - **TypeError** - 首个输入的数据类型不为Tensor、`dst` 不是int，或 `group` 不是str。
        - **RuntimeError** - 目标设备无效、后端无效，或分布式初始化失败。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

        该样例需要在多卡环境下运行。
