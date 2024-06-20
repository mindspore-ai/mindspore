mindspore.ops.CollectiveGather
==============================

.. py:class:: mindspore.ops.CollectiveGather(dest_rank, group=GlobalComm.WORLD_COMM_GROUP)

    对通信组的输入张量进行聚合。操作会将每张卡的输入Tensor的第0维度上进行聚合，发送到对应卡上。

    .. note::
        只有目标为dest_rank的进程(全局的进程编号)才会收到聚合操作后的输出。其他进程只得到一个形状为[1]的张量，且该张量没有数学意义。


    参数：
        - **dest_rank** (int) - 表示发送目标的进程编号。只有该进程会接收汇聚张量。
        - **group** (str，可选) - 表示通信域。默认值： ``GlobalComm.WORLD_COMM_GROUP`` 。

    输入：
        - **input_x** (Tensor) - 输入待聚合的Tensor，Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。

    输出：
        Tensor，Tensor第0维等于各输入数据第0维的累加，其他shape维度相同
        即 :math:`(\sum x_1, x_2, ..., x_R)` 。

    异常：
        - **TypeError** - 首个输入的数据类型不为Tensor，`op` 或 `group` 不是str。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。
        - **ValueError** - 调用进程的rank id大于本通信组的rank大小。

    样例：

    .. note::
        .. include:: mindspore.ops.comm_note.rst

        该样例需要在2卡环境下运行。
    
    教程样例：
        - `分布式集合通信原语 - CollectiveGather
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/ops/communicate_ops.html#collectivegather>`_
