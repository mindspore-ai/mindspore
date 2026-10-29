mindspore.ops.AlltoAllV
=======================

.. py:class:: mindspore.ops.AlltoAllV(group=GlobalComm.WORLD_COMM_GROUP, block_size=1)

    相对AlltoAll来说，AlltoAllV算子支持不等分的切分和聚合。

    .. note::
        只支持一维的输入，使用该接口前需要将输入数据展开成一维。

    参数：
        - **group** (str，可选) - AlltoAll的通信域。默认值： ``GlobalComm.WORLD_COMM_GROUP`` ，Ascend平台表示为 ``"hccl_world_group"`` 。
        - **block_size** (int，可选) - 通过 `send_numel_list` 和 `recv_numel_list` 切分和聚合数据量的基本单位。默认值： ``1`` 。

    输入：
        - **input_x** (Tensor) - 一维待分发的张量, shape为 :math:`(x_1)`。
        - **send_numel_list** (Union[tuple[int], list[int], Tensor]) - 分发给每张卡的数据量。实际分发数据量为 :math:`(send\_numel\_list * block\_size * input\_x.dtype)` 。
        - **recv_numel_list** (Union[tuple[int], list[int], Tensor]) - 从每张卡聚合的数据量。实际聚合数据量为 :math:`(recv\_numel\_list * block\_size * input\_x.dtype)` 。

    输出：
        Tensor，从每张卡上聚合的一维数据结果。如果结果为空，则返回空张量，且张量数值无意义。

    样例：

    .. note::
        .. include:: mindspore.ops.comm_note.rst

        该样例需要在2卡环境下运行。

    教程样例：
        - `分布式集合通信原语 - AlltoAllV
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/ops/communicate_ops.html#alltoallv>`_