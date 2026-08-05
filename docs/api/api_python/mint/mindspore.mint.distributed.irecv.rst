mindspore.mint.distributed.irecv
=====================================

.. py:function:: mindspore.mint.distributed.irecv(tensor, src=0, group=None, tag=0)

    异步从指定源进程接收tensor。

    .. note::
        - 当前仅支持PyNative模式，不支持Graph模式。
        - 在MCCL通信组场景， :func:`mindspore.mint.distributed.isend` 和 :func:`mindspore.mint.distributed.irecv` 非强同步执行。

    参数：
        - **tensor** (Tensor) - 接收发送方数据存入tensor，shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **src** (int，可选) - 表示发送源的进程编号。只会接收来自源进程的tensor。默认值： ``0``。
        - **group** (str，可选) - 通信组名称。默认值： ``None`` ，即在Ascend平台表示为 ``"hccl_world_group"`` 。
        - **tag** (int，可选) - 用于区分发送、接收消息的标签。该消息将接收来自相同 `tag` 的Send发送的tensor。默认值： ``0`` 。当前为预留参数。

    返回：
        CommHandle，异步工作句柄。

    异常：
        - **TypeError** - `tensor` 不是Tensor， `src` 不是int或 `group` 不是str。
        - **ValueError** - 如果该进程的rank id 大于通信组的rank size。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
