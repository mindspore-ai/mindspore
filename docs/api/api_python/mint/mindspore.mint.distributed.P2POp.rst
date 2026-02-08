mindspore.mint.distributed.P2POp
=====================================

.. py:class:: mindspore.mint.distributed.P2POp(op, tensor, peer, group=None, tag=0)

    用于存放关于'isend'、'irecv'相关的信息， 并用于 `batch_isend_irecv` 接口的入参。

    .. note::
        - `tensor` 当 `op` 入参为'irecv'时，入参最后的结果会原地修改。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **op** (Union[str, function]) - 对于字符串类型，只允许'isend'和'irecv'。 对于函数类型，只允许 ``distributed.isend`` 和 ``distributed.irecv`` 函数。
        - **tensor** (Tensor) - 用于发送或接收的张量。
        - **peer** (int) - 发送或接收的远程设备的全局编号。
        - **group** (str，可选) - 工作的通信组，默认值： ``None`` （即Ascend平台为 ``"hccl_world_group"``）。
        - **tag** (int，可选) - 当前暂不支持。默认值： ``0`` 。

    返回：
        `P2POp` 对象。

    异常：
        - **TypeError** - 当 `op` 不是与'isend'和'irecv'相关的字符串或函数。
        - **TypeError** - 当 `tensor` 不是张量， `peer` 不是int。
        - **NotImplementedError** - 当 `tag` 入参不为0。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst
