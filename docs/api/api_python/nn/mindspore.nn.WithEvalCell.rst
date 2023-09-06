mindspore.nn.WithEvalCell
=========================

.. py:class:: mindspore.nn.WithEvalCell(network, loss_fn, add_cast_fp32=False)

    封装前向网络和损失函数。
    返回用于计算评估指标的损失函数值、前向输出和标签。

    参数：
        - **network** (Cell) - 前向网络。
        - **loss_fn** (Cell) - 损失函数。
        - **add_cast_fp32** (bool) - 是否将数据类型调整为float32。默认值： ``False`` 。

    输入：
        - **data** (Tensor) - shape为 :math:`(N, \ldots)` 的Tensor。
        - **label** (Tensor) - shape为 :math:`(N, \ldots)` 的Tensor。

    输出：
        Tuple(Tensor)，包括标量损失函数、shape为 :math:`(N, \ldots)` 的网络输出和shape为 :math:`(N, \ldots)` 的标签。

    异常：
        - **TypeError** - `add_cast_fp32` 不是bool。
