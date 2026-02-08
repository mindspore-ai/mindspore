mindspore.mint.distributed.irecv
=====================================

.. py:function:: mindspore.mint.distributed.irecv(tensor, src=0, group=None, tag=0)

    异步接收Tensor到指定线程。

    .. note::
        当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **tensor** (Tensor) - 接收发送方数据存入Tensor。Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **src** (int，可选) - 表示发送源的进程编号。只会接收来自源进程的Tensor。默认值： ``0``。
        - **group** (str，可选) - 通信组名称，如果为 ``None`` ，Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。
        - **tag** (int，可选) - 用于区分发送、接收消息的标签。该消息将接收来自相同 `tag` 的Send发送的Tensor。默认值： ``0`` 。当前为预留参数。

    返回：
        CommHandle，CommHandle是一个异步工作句柄。

    异常：
        - **TypeError** - `tensor` 不是Tensor， `src` 不是int或 `group` 不是str。
        - **ValueError** - 如果该线程的rank id 大于通信组的rank size。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
