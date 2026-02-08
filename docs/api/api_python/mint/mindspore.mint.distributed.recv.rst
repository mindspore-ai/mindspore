mindspore.mint.distributed.recv
=====================================

.. py:function:: mindspore.mint.distributed.recv(tensor, src=0, group=None, tag=0)

    同步接收Tensor到指定线程。

    .. note::
        当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **tensor** (Tensor) - 输入Tensor。Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。
          输入的 `tensor` 的shape和dtype将用于接收Tensor，但 `tensor` 的数据值不起作用。
        - **src** (int，可选) - 表示发送源的进程编号。只会接收来自源进程的Tensor。默认值： ``0`` 。
        - **group** (str，可选) - 通信组名称，如果为 ``None`` ，Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。
        - **tag** (int，可选) - 用于区分发送、接收消息的标签。该消息将被接收来自相同 `tag` 的Send发送的张量。默认值： ``0`` 。当前为预留参数。

    返回：
        int，如果成功接收，返回值为 ``0`` 。

    异常：
        - **TypeError** - `tensor` 不是Tensor， `src` 不是int，或 `group` 不是str。
        - **ValueError** - 如果该线程的rank id大于通信组的rank size。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
