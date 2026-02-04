mindspore.communication.comm_func.batch_isend_irecv
=================================================================================

.. py:function:: mindspore.communication.comm_func.batch_isend_irecv(p2p_op_list)

    异步地发送和接收张量。

    .. note::
        - 不同设备中， `p2p_op_list` 中的 `P2POp` 的 ``"isend"`` 和 ``"irecv"`` 应该互相匹配。
        - `p2p_op_list` 中的 `P2POp` 应该使用同一个通信组。
        - 暂不支持 `p2p_op_list` 中的 `P2POp` 含有 `tag` 入参。
        - `p2p_op_list` 中的 `P2POp` 的 `tensor` 的值不会被最后的结果原地修改。
        - 仅支持PyNative模式，目前不支持Graph模式。

    参数：
        - **p2p_op_list** (list[P2POp]) - 包含 `P2POp` 类型对象的列表。 `P2POp` 指的是 :class:`mindspore.communication.comm_func.P2POp`。

    返回：
        Tuple(Tensor)，根据 `p2p_op_list` 中的 `P2POp` 的发送/接收顺序，得到的接收张量元组。
        
        - 当 `P2POp` 为发送时，相应位置的结果是没有意义的张量。
        - 当 `P2POp` 为接收时，相应位置的结果是从其他设备接收到的张量。

    异常：
        - **TypeError** - `p2p_op_list` 中不全是 `P2POp` 类型。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

        该样例需要在2卡环境下运行。
