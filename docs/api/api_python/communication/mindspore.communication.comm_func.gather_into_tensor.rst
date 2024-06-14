mindspore.communication.comm_func.gather_into_tensor
====================================================

.. py:function:: mindspore.communication.comm_func.gather_into_tensor(tensor, dst=0, group=GlobalComm.WORLD_COMM_GROUP)

    对通信组的输入张量进行聚合。操作会将每张卡的输入Tensor的第0维度上进行聚合，发送到对应卡上。

    .. note::
        只有目标为dst的进程(全局的进程编号)才会收到聚合操作后的输出。其他进程只得到一个形状为[1]的张量，且该张量没有数学意义。
        当前支持PyNative模式，不支持Graph模式。


    参数：
        - **tensor** (Tensor) - 输入待聚合的Tensor。Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **dst** (int，可选) - 表示发送源的进程编号。只有该进程会接收聚合后的张量。默认值：0。
        - **group** (str，可选) - 表示通信域。默认值： ``GlobalComm.WORLD_COMM_GROUP`` 。

    返回：
        Tensor，即 :math:`(\sum x_1, x_2, ..., x_R)`。Tensor第0维等于输入数据第0维求和，其他shape相同。

    异常：
        - **TypeError** - 首个输入的数据类型不为Tensor，`op` 或 `group` 不是str。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

        该样例需要在多卡环境下运行。
