mindspore.ops.communication.reduce_scatter_tensor_uneven
==========================================================

.. py:function:: mindspore.ops.communication.reduce_scatter_tensor_uneven(output, input, input_split_sizes=None, op=ReduceOp.SUM, group=None, async_op=False)

    根据 `input_split_sizes` 对指定通信组中的张量进行归约操作，并分散到输出张量。

    .. note::
        - 输入张量在所有进程之间必须具有相同的shape和格式。
        - 输入张量的第一维应等于 `input_split_sizes` 的总和。

    参数：
        - **output** (Tensor) - 输出张量，与 `input` 具有相同的数据类型，shape为 :math:`(input\_split\_sizes[rank], *)`，其中rank是设备的本地rank id。
        - **input** (Tensor) - 待归约和分散的输入张量，期望shape为 :math:`(N, *)`，其中 `*` 表示任意数量的额外维度。N必须等于所有rank的 `input_split_sizes` 的总和。
        - **input_split_sizes** (list[int], 可选) - 指定如何分割输入张量第一维的列表。如果为 ``None``，则根据组大小平均分割。默认值： ``None``。
        - **op** (str，可选) - 用于逐元素归约的操作，ReduceOp之一：'SUM'、'MIN'、'MAX'。默认值： ``ReduceOp.SUM``。
        - **group** (str，可选) - 通信组名称，如果为 ``None`` ，Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False``。

    返回：
        CommHandle，若 `async_op` 是True，CommHandle是一个异步工作句柄。若 `async_op` 是False，CommHandle将返回None。

    异常：
        - **ValueError** - 如果 `output` 的shape与 `input_split_sizes` 的约束不匹配。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
