mindspore.mint.distributed.batch_isend_irecv
================================================

.. py:function:: mindspore.mint.distributed.batch_isend_irecv(p2p_op_list)

    异步地发送和接收张量。

    .. note::
        - 不同设备中， `p2p_op_list` 中的 `P2POp` 的 ``"isend`` 和 ``"irecv"`` 应该互相匹配。
        - `p2p_op_list` 中的 `P2POp` 应该使用同一个通信组。
        - 暂不支持 `p2p_op_list` 中的 `P2POp` 含有 `tag` 入参。
        - `p2p_op_list` 中的 `P2POp` 的 `tensor` 的值不会被最后的结果原地修改。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **p2p_op_list** (list[P2POp]) - 包含 `P2POp` 类型对象的列表。 `P2POp` 指的是 :class:`mindspore.mint.distributed.P2POp`。

    返回：
        list[CommHandle]，当前list元素为1，CommHandle是一个异步工作句柄。

    异常：
        - **TypeError** - `p2p_op_list` 为空，或 `p2p_op_list` 中不全是 `P2POp` 类型。
        - **TypeError** - 通信组名在 `p2p_op_list` 不一致。
        - **TypeError** -  `tensor` 在 `p2p_op_list` 中不为Tensor。
        - **TypeError** -  `op` 在 `p2p_op_list` 中不是'isend'或'irecv'。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
