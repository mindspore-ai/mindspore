mindspore.nn.TrainOneStepCell
=============================

.. py:class:: mindspore.nn.TrainOneStepCell(network, optimizer, sens=None, return_grad=False)

    训练网络封装类。

    封装 `network` 和 `optimizer` 。构建一个输入'\*inputs'的用于训练的Cell。
    执行函数 `construct` 中会构建反向图以更新网络参数。支持不同的并行训练模式。

    参数：
        - **network** (Cell) - 训练网络。只支持单输出网络。
        - **optimizer** (Union[Cell]) - 用于更新网络参数的优化器。
        - **sens** (numbers.Number) - 反向传播的输入，缩放系数。默认值为 ``None`` ，取 ``1.0`` 。
        - **return_grad** (bool) - 是否返回梯度，若为 ``True`` ，则会在返回loss的同时以字典的形式返回梯度，字典的key为梯度对应的参数名，value为梯度值。默认值为 ``False`` 。

    输入：
        - **\*inputs** (Tuple(Tensor)) - shape为 :math:`(N, \ldots)` 的Tensor组成的元组。

    输出：
        Tensor，损失函数值，其shape通常为 :math:`()` 。

    异常：
        - **TypeError** - `sens` 不是numbers.Number。
