mindspore.ops.communication.send
==================================

.. py:function:: mindspore.ops.communication.send(tensor, dst=0, group=None, tag=0)

    将张量发送到指定的目标rank。

    .. note::
        当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **tensor** (Tensor) - 要发送的张量。
        - **dst** (int, 可选) - 标识目标rank（全局rank）的整数。默认值： ``0``。
        - **group** (str，可选) - 通信组名称，如果为 ``None`` ，Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。
        - **tag** (int, 可选) - 标识发送/接收消息标签的整数。消息将被具有相同"tag"的Receive操作接收。默认值： ``0``。当前为预留参数。

    异常：
        - **TypeError** - `tensor` 不是Tensor，`dst` 不是int或 `group` 不是str。
        - **ValueError** - 如果 `dst` 进程rank id与当前进程相同。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
