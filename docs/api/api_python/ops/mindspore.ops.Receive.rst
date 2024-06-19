mindspore.ops.Receive
=====================

.. py:class:: mindspore.ops.Receive(sr_tag, src_rank, shape, dtype, group=GlobalComm.WORLD_COMM_GROUP)

    接收来自 `src_rank` 线程的张量。

    .. note::
        Send 和 Receive 算子需组合使用，且有同一个 `sr_tag` 。

    参数：
        - **sr_tag** (int) - 用于区分发送、接收消息的标签。该消息将被接收来自相同 `sr_tag` 的Send发送的张量。
        - **src_rank** (int) - 表示发送源的进程编号。只会接收来自源进程的张量。
        - **shape** (list[int]) - 表示发送源的张量形状。
        - **dtype** (Type) - 表示发送源的张量类型。所支持的类型有：int8, int16, int32, float16, float32。
        - **group** (str，可选) - 表示通信域。默认值： ``GlobalComm.WORLD_COMM_GROUP`` 。

    输出：
        - Tensor - Tensor的shape与输入的shape相同。

    异常：
        - **TypeError** - src_rank不是int或group不是str。
        - **ValueError** - 如果该线程的rank id 大于通信组的rank size。