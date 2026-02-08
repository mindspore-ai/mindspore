mindspore.communication.comm_func.irecv
=======================================

.. py:function:: mindspore.communication.comm_func.irecv(tensor, src=0, group=GlobalComm.WORLD_COMM_GROUP, tag=0)

    异步接收Tensor到指定线程。

    .. note::
        - Send 和 Receive 算子需组合使用，且有同一个 `tag`。
        - 输入的 `tensor` 的shape和dtype将用于接收Tensor，但 `tensor` 的数据值不起作用。
        - 当前支持PyNative模式，不支持Graph模式。

    参数：
        - **tensor** (Tensor) - 输入Tensor，shape为 :math:`(x_1, x_2, ..., x_R)` 。
          输入的 `tensor` 的shape和dtype将用于接收Tensor，但 `tensor` 的数据值不起作用。
        - **src** (int, 可选) - 表示发送源的进程编号。只会接收来自该源进程的Tensor。默认值：``0``。
        - **group** (str, 可选) - 工作的通信组。默认值： ``GlobalComm.WORLD_COMM_GROUP`` （即Ascend平台为 ``"hccl_world_group"`` ，GPU平台为 ``"nccl_world_group"`` ）。
        - **tag** (int, 可选) - 用于区分发送、接收消息的标签。只会接收来自相同 `tag` 的Send算子发送的Tensor。默认值：``0``。

    返回：
        Tuple(Tensor, CommHandle)，输出Tensor的shape为 :math:`(x_1, x_2, ..., x_R)`。
        CommHandle是一个异步工作句柄。

    异常：
        - **TypeError** - `src` 不是int，或 `group` 不是str。
        - **ValueError** - 该进程的rank id大于通信组的rank size。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

        该样例需要在2卡环境下运行。
