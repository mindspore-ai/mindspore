mindspore.ops.AlltoAllVC
========================

.. py:class:: mindspore.ops.AlltoAllVC(group=GlobalComm.WORLD_COMM_GROUP, block_size=1, transpose=False)

    AlltoAllVC通过输入参数 `send_count_matrix` 传入所有rank的收发参数。相对AlltoAllV来说，AlltoAllVC不需要汇聚所有rank的收发参数，因此性能更优。

    .. note::
        只支持一维的输入，使用该接口前需要将输入数据展开成一维。

    参数：
        - **group** (str，可选) - AlltoAll的通信域。默认值： ``GlobalComm.WORLD_COMM_GROUP`` ，在Ascend平台中表示为 ``"hccl_world_group"`` 。
        - **block_size** (int，可选) - 通过 `send_count_matrix` 切分和聚合数据量的基本单位。默认值： ``1`` 。
        - **transpose** (bool，可选) - 表示 `send_count_matrix` 是否需要做转置运算，该参数在反向计算场景下使用。默认值： ``False`` 。

    输入：
        - **input_x** (Tensor) - 一维待分发的张量，shape为 :math:`(x_1)`。
        - **send_count_matrix** (Union[list[int], Tensor]) - 所有rank的收发参数， :math:`\text{send_count_matrix}[i*\text{rank_size}+j]` 表示rank i发给rank j的数据量，基本单位是Tensor的数据类型。其中， `rank_size` 表示通信组大小。

    输出：
        Tensor，从每张卡上聚合的一维数据结果。如果结果为空，则返回空张量，且值无意义。

    样例：

    .. note::
        .. include:: mindspore.ops.comm_note.rst

        该样例需要在2卡环境下运行。
    
    教程样例：
        - `分布式集合通信原语 - AlltoAllVC
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/ops/communicate_ops.html#alltoallvc>`_
