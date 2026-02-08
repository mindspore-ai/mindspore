mindspore.communication.comm_func.P2POp
=========================================

.. py:class:: mindspore.communication.comm_func.P2POp(op, tensor, peer, group=None, tag=0, *, recv_dtype=None)

    用于存放关于'isend'、'irecv'相关的信息，并用于 `batch_isend_irecv` 接口的入参。

    .. note::
        - 当 `op` 入参为'irecv'时，`tensor` 入参允许不传入张量，可以只传入接收张量的形状。
        - `tensor` 入参不会被最后的结果原地修改。

    参数：
        - **op** (Union[str, function]) - 对于字符串类型，只允许'isend'和'irecv'。对于函数类型，只允许 ``comm_func.isend`` 和 ``comm_func.irecv`` 函数。
        - **tensor** (Union[Tensor, Tuple(int)]) - 用于发送或接收的张量。如果 `op` 是'irecv'，可以传入接收张量的形状。
        - **peer** (int) - 发送或接收的远程设备的全局编号。
        - **group** (str，可选) - 工作的通信组。默认值：``None``，表示在Ascend上使用 ``hccl_world_group``，在GPU上使用 ``nccl_world_group``。
        - **tag** (int，可选) - 当前暂不支持。默认值： ``0``。

    关键字参数：
        - **recv_dtype** (mindspore.dtype，可选) - 表示接收张量的数据类型。当 `tensor` 传入的是张量的形状时，该入参必须要配置。默认值： ``None``。

    返回：
        `P2POp` 对象。

    异常：
        - **ValueError** - 当 `op` 不是与'isend'、'irecv'相关的字符串或函数。
        - **TypeError** - 当 `tensor` 不是张量或者元组类型。
        - **NotImplementedError** - 当 `tag` 入参不为 ``0``。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

        该样例需要在2卡环境下运行。
