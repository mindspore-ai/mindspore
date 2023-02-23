mindspore.nn.TrainOneStepCell
=============================

.. py:class:: mindspore.nn.TrainOneStepCell(network, optimizer, sens=1.0)

    训练网络封装类。

    封装 `network` 和 `optimizer` 。构建一个输入'\*inputs'的用于训练的Cell。
    执行函数 `construct` 中会构建反向图以更新网络参数。支持不同的并行训练模式。

    参数：
        - **network** (Cell) - 训练网络。只支持单输出网络。
        - **optimizer** (Union[Cell]) - 用于更新网络参数的优化器。
        - **sens** (numbers.Number) - 反向传播的输入，缩放系数。默认值为1.0。

    输入：
        - **\*inputs** (Tuple(Tensor)) - shape为 :math:`(N, \ldots)` 的Tensor组成的元组。

    输出：
        Tensor，损失函数值，其shape通常为 :math:`()` 。

    异常：
        - **TypeError** - `sens` 不是numbers.Number。
