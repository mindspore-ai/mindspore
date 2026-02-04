mindspore.mint.distributed.send
=====================================

.. py:function:: mindspore.mint.distributed.send(tensor, dst=0, group=None, tag=0)

    同步发送张量到指定线程。

    .. note::
        当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **tensor** (Tensor) - 输入Tensor。Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **dst** (int，可选) - 表示发送目标的进程编号。只有目标进程会收到张量。默认值： ``0`` 。
        - **group** (str，可选) - 通信组名称，如果为 ``None`` ，Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。
        - **tag** (int，可选) - 用于区分发送、接收消息的标签。该消息将被拥有相同 `tag` 的Receive接收。默认值： ``0`` 。当前为预留参数。

    异常：
        - **TypeError** - 输入 `tensor` 的数据类型不为Tensor， dst不是int或group不是str。
        - **ValueError** - 如果发送线程和目的线程号相同。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
