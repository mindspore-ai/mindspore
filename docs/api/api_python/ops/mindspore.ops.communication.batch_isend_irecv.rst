mindspore.ops.communication.batch_isend_irecv
===============================================

.. py:function:: mindspore.ops.communication.batch_isend_irecv(p2p_op_list)

    批量异步发送和接收张量。

    .. note::
        - `p2p_op_list` 中 `P2POp` 的'isend'和'irecv'在rank之间需要相互匹配。
        - `p2p_op_list` 中的 `P2POp` 只能使用相同的通信组。
        - `p2p_op_list` 中 `P2POp` 的 `tag` 暂不支持。
        - `p2p_op_list` 中 `P2POp` 的 `tensor` 不会被结果原地修改。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **p2p_op_list** (list[P2POp]) - 包含 `P2POp` 的列表。`P2POp` 的类型为 :class:`mindspore.ops.communication.P2POp`

    返回：
        list[CommHandle]，CommHandle是一个异步工作句柄，目前仅支持一个包装句柄。

    异常：
        - **TypeError** - 如果 `p2p_op_list` 为空或 `p2p_op_list` 不全是 `P2POp` 类型。
        - **TypeError** - `p2p_op_list` 中的组名不一致。
        - **TypeError** - `p2p_op_list` 中的 `tensor` 不是Tensor。
        - **TypeError** - `p2p_op_list` 中的 `op` 不是isend或irecv。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
