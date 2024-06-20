mindspore.ops.Reduce
====================

.. py:class:: mindspore.ops.Reduce(dest_rank, op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP)

    规约指定通信组中的张量，并将规约结果发送到目标为dest_rank的进程中，返回发送到目标进程的张量。

    .. note::
        只有目标为dest_rank的进程(通信组的本地进程编号)才会收到规约操作后的输出。
        当前支持Pynative和Graph模式。但Graph模式只支持图编译等级为O0的场景。
        其他进程只得到一个形状为[1]的张量，且该张量没有数学意义。

    参数：
        - **dest_rank** (int) - 指定接收输出的目标进程编号(通信组的本地进程编号)，只有该进程会接收规约操作后的输出结果。
        - **op** (str，可选) - 规约的具体操作。如 ``"sum"`` 、 ``"prod"`` 、 ``"max"`` 、和 ``"min"`` 。默认值： ``ReduceOp.SUM`` 。
        - **group** (str，可选) - 工作的通信组。默认值：``GlobalComm.WORLD_COMM_GROUP`` （即Ascend平台为 ``"hccl_world_group"`` ，GPU平台为 ``"nccl_world_group"`` ）。

    输入：
        - **input_x** (Tensor) - Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。

    输出：
        Tensor，返回规约操作后，目标进程的tensor。数据类型与输入的 `tensor` 一致，shape为 :math:`(x_1, x_2, ..., x_R)`。

    异常：
        - **TypeError** - 首个输入的数据类型不为Tensor，`op` 和 `group` 不是字符串。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.ops.comm_note.rst

        该样例需要在4卡环境下运行。
    
    教程样例：
        - `分布式集合通信原语 - Reduce
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/ops/communicate_ops.html#reduce>`_
