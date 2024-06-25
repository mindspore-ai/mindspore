mindspore.ops.Send
==================

.. py:class:: mindspore.ops.Send(sr_tag, dest_rank, group=GlobalComm.WORLD_COMM_GROUP, group_back=GlobalComm.WORLD_COMM_GROUP)

    发送张量到指定线程。

    .. note::
        Send 和 Receive 算子需组合使用，且有同一个 `sr_tag`。

    参数：
        - **sr_tag** (int) - 用于区分发送、接收消息的标签。该消息将被拥有相同 `sr_tag` 的Receive接收。
        - **dest_rank** (int) - 表示发送目标的进程编号。只有目标进程会收到张量。
        - **group** (str，可选) - 表示通信域。默认值： ``GlobalComm.WORLD_COMM_GROUP`` 。
        - **group_back** (str，可选) - 表示计算反向传播时的通信域。默认值： ``GlobalComm.WORLD_COMM_GROUP`` 。

    输入：
        - **input_x** (Tensor) - 输入待发送的Tensor，Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。

    异常：
        - **TypeError** - dest_rank不是int或group不是str。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。
        - **ValueError** - 如果该线程的rank id 大于通信组的rank size。

    样例：

    .. note::
        .. include:: mindspore.ops.comm_note.rst

        该样例需要在2卡环境下运行。
    
    教程样例：
        - `分布式集合通信原语 - Send
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/ops/communicate_ops.html#send>`_
