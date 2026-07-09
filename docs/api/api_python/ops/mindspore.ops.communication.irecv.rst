mindspore.ops.communication.irecv
===================================

.. py:function:: mindspore.ops.communication.irecv(tensor, src=0, group=None, tag=0)

    异步从src接收张量。

    .. note::
        - 当前仅支持PyNative模式，不支持Graph模式。
        - 在MCCL通信组场景， :func:`mindspore.ops.communication.isend` 和 :func:`mindspore.ops.communication.irecv` 非强同步执行。

    参数：
        - **tensor** (Tensor) - 如果函数以in-place模式运行，则用于填充接收数据的张量。否则，表示用于接收张量的shape和dtype，但输入 `tensor` 的值不会生效。
        - **src** (int, 可选) - 标识源rank（全局rank）的整数。默认值： ``0``。
        - **group** (str，可选) - 通信组名称。默认值： ``None`` ，即Ascend平台表示为 ``"hccl_world_group"`` 。
        - **tag** (int, 可选) - 标识发送/接收消息标签的整数。消息将被具有相同 `tag` 的Send操作接收。默认值： ``0``。当前为预留参数。

    返回：
        - 若函数以in-place模式运行，返回CommHandle。
        - 若函数以非in-place模式运行，返回Tuple(Tensor, CommHandle)。第一个元素存储输出结果，第二个元素是CommHandle。

    异常：
        - **TypeError** - `tensor` 的类型不是Tensor，`src` 不是int或 `group` 不是str。
        - **ValueError** - 如果进程的rank ID大于通信组的rank size。

    样例：

    .. note::
        .. include:: ../mint/mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
