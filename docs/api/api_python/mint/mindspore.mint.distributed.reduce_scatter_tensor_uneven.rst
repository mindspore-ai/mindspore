mindspore.mint.distributed.reduce_scatter_tensor_uneven
=======================================================

.. py:function:: mindspore.mint.distributed.reduce_scatter_tensor_uneven(output, input, input_split_sizes=None, op=ReduceOp.SUM, group=None, async_op=False)

    在指定通信组中执行归约分发操作，根据 `input_split_sizes` 将归约后的张量分散到各rank的输出张量中。

    .. note::
        - 输入张量在所有进程中的shape和格式必须一致。
        - 输出张量的第一个维度尺寸应该等于 `input_split_sizes` 的所有值之和。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **output** (Tensor) - 输出张量，与输入张量具有相同数据类型，shape为 :math:`(input\_split\_sizes[rank], *)`，其中rank是当前的设备的id。
        - **input** (Tensor) - 待归约分发的输入张量，shape为 :math:`(N, *)`，`*` 表示任意数量的附加维度，N应为各rank的 `input_split_sizes` 值之和。
        - **input_split_sizes** (list[int], 可选) - 输入张量在第一个维度的切分尺寸列表。当为None时，将按通信组大小对输入张量进行均分。默认值： ``None``。
        - **op** (str, 可选) - 归约的具体操作。可选值： ``"sum"`` 、 ``"max"`` 、 ``"min"`` 。默认值： ``ReduceOp.SUM``。
        - **group** (str，可选) - 通信组名称，如果为 ``None``，Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None``。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False``。

    返回：
        CommHandle。若 `async_op` 是True，CommHandle是一个异步工作句柄；若 `async_op` 是False，CommHandle将返回None。

    异常：
        - **ValueError** - 如果 `output` 的shape与 `input_split_sizes` 的值不满足约束。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
