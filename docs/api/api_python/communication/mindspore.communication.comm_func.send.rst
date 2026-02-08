mindspore.communication.comm_func.send
=======================================

.. py:function:: mindspore.communication.comm_func.send(tensor, dst=0, group=GlobalComm.WORLD_COMM_GROUP, tag=0)

    同步发送Tensor到指定进程。

    .. note::
        - Send 和 Receive 算子需组合使用，且有同一个 `tag`。
        - 当前支持PyNative模式，不支持Graph模式。

    参数：
        - **tensor** (Tensor) - 待发送的Tensor，shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **dst** (int, 可选) - 表示发送目标的进程编号。只有目标进程会收到Tensor。默认值：``0``。
        - **group** (str, 可选) - 工作的通信组。默认值： ``GlobalComm.WORLD_COMM_GROUP`` （即Ascend平台为 ``"hccl_world_group"`` ，GPU平台为 ``"nccl_world_group"`` ）。
        - **tag** (int, 可选) - 用于区分发送、接收消息的标签。该消息将被拥有相同 `tag` 的Receive接收。默认值：``0``。

    异常：
        - **TypeError** - `dst` 不是int，或 `group` 不是str。
        - **ValueError** - 该进程的rank id大于通信组的rank size。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

        该样例需要在2卡环境下运行。
