mindspore.ops.ReduceScatterV
============================

.. py:class:: mindspore.ops.ReduceScatterV(op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP)

    归约并且分发指定通信组中不均匀的张量，返回分发后的张量。

    .. note::
        只支持一维的输入，使用该接口前需要将输入数据展开成一维。

    参数：
        - **op** (str, 可选) - 指定用于元素的归约操作，如SUM、MIN和MAX，当前不支持PROD。默认值： ``ReduceOp.SUM`` 。
        - **group** (str，可选) - 工作的通信组，默认值： ``GlobalComm.WORLD_COMM_GROUP`` （即Ascend平台为 ``"hccl_world_group"`` ，GPU平台为 ``"nccl_world_group"`` ）。

    输入：
        - **input_x** (Tensor) - 一维待分发的张量，shape为 :math:`(x_1)`。
        - **input_split_sizes** (Union[tuple[int], list[int], Tensor]) - 一维张量，所有rank的接收数据量列表，基本单位是Tensor的数据类型。该数值未作校验，由用户保障其正确性。

    输出：
        Tensor，从每张卡上归约并且分发的一维数据结果。如果结果为空，则返回空张量，且值无意义。

    异常：
        - **RuntimeError** - 目标设备无效、后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.ops.comm_note.rst

        该样例需要在2卡环境下运行。

    教程样例：
        - `分布式集合通信原语 - ReduceScatterV
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/ops/communicate_ops.html#reducescatterv>`_
