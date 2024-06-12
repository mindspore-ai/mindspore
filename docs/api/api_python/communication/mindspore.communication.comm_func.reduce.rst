mindspore.communication.comm_func.reduce
========================================

.. py:function:: mindspore.communication.comm_func.reduce(tensor, dst, op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP)

    规约指定通信组中的张量，并将规约结果发送到目标为dst的进程(全局的进程编号)中，返回发送到目标进程的张量。

    .. note::
        只有目标为dst的进程(全局的进程编号)才会收到规约操作后的输出。
        当前支持PyNative模式，不支持Graph模式。
        其他进程只得到一个形状为[1]的张量，且该张量没有数学意义。

    参数：
        - **tensor** (Tensor) - 输入待规约的Tensor，Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **dst** (int) - 指定接收输出的目标进程编号，只有该进程会接收规约操作后的输出结果。
        - **op** (str, 可选) - 规约的具体操作。如 ``"sum"`` 、 ``"prod"`` 、 ``"max"`` 、和 ``"min"`` 。默认值： ``ReduceOp.SUM`` 。
        - **group** (str，可选) - 工作的通信组，默认值： ``GlobalComm.WORLD_COMM_GROUP`` （即Ascend平台为 ``"hccl_world_group"`` ，GPU平台为 ``"nccl_world_group"`` ）。

    返回：
        Tensor，数据类型与输入的 `tensor` 一致，shape为 :math:`(x_1, x_2, ..., x_R)`。

    异常：
        - **TypeError** - 首个输入的数据类型不为Tensor，`op` 和 `group` 不是字符串。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

        该样例需要在4卡环境下运行。
