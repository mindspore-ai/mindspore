mindspore.mint.distributed.isend
=====================================

.. py:function:: mindspore.mint.distributed.isend(tensor, dst=0, group=None, tag=0)

    异步发送张量到指定线程。

    .. note::
        当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **tensor** (Tensor) - 输入发送Tensor。
        - **dst** (int，可选) - 表示发送目标的进程编号。只有目标进程会收到张量。默认值： ``0``。
        - **group** (str，可选) - 通信组名称，如果为 ``None`` ，Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。
        - **tag** (int，可选) - 用于区分发送、接收消息的标签。该消息将被拥有相同 `tag` 的Receive接收。默认值： ``0``。当前为预留参数。

    返回：
        CommHandle，CommHandle是一个异步工作句柄。

    异常：
        - **TypeError** -  `tensor` 不是Tensor， `dst` 不是int或 `group` 不是str。
        - **ValueError** - 如果发送线程和目的线程号相同。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
