mindspore.communication.comm_func.all_to_all_single
===================================================

.. py:function:: mindspore.communication.comm_func.all_to_all_single(tensor, output_split_sizes=None, input_split_sizes=None, group=GlobalComm.WORLD_COMM_GROUP)

    切分张量并散射到通信域内多个进程。

    将输入数据在第0维切分成特定的块数（blocks），并按顺序散射。一般有三个阶段：

    - 准备阶段：入参 `input_split_sizes`、`output_split_sizes` 的校验，并计算切分块数( `split_count` )。
    - 散射阶段：在每个进程上，操作数沿着第0维拆分为 `split_count` 个块（blocks），且散射到指定的rank上，例如，第i块被发送到第i个rank上。
    - 聚合阶段：每个rank沿着第0维拼接接收到的数据。

    该算子暂不支持不均匀切分并分发，`input_split_sizes` 和 `output_split_sizes` 中元素必须相同。

    .. note::
        聚合阶段，所有进程中的Tensor必须具有相同的shape和格式。
        当前支持PyNative模式，不支持Graph模式。

    参数：
        - **tensor** (Tensor) - 输入Tensor。Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **output_split_sizes** (Tuple[int], 可选) - 输出数据的第0维切分大小。该算子暂不支持不均匀切分并分发，`output_split_sizes` 中元素必须相同。默认值：``None``。
        - **input_split_sizes** (Tuple[int], 可选) - 输入数据的第0维切分大小。该算子暂不支持不均匀切分并分发，`output_split_sizes` 中元素必须相同。默认值：``None``。
        - **group** (str, 可选) - 表示通信域。默认值： ``GlobalComm.WORLD_COMM_GROUP`` 。

    输出：
        Tensor，设输入的shape是 :math:`(x_1, x_2, ..., x_R)`，则输出的shape为 :math:`(y_1, x_2, ..., x_R)`。

    异常：
        - **TypeError** - 如果 `group` 不是字符串。
        - **TypeError** - 如果 `output_split_sizes` 或 `input_split_sizes` 中元素不一致。
        - **TypeError** - 如果 `output_split_sizes` 或 `input_split_sizes` 的求和不等于tensor的第0个维度值。
        - **ValueError** - 如果 `split_count` 无法被 `rank_size` 整除。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

        该样例需要在多卡环境下运行。
