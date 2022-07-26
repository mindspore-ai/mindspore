mindspore.ops.ger
=================

.. py:function:: mindspore.ops.ger(x1, x2)

    计算输入一维Tensor `x1` 与 `x2` 的外积。如果 `x1` shape为 :math:`(m,)` ，`x2` shape为 :math:`(n,)` ，则输出的shape为 :math:`(m, n)` 。

    .. note::
        Ascend不支持float64数据格式的输入。

    参数：
        - **x1** (Tensor) - 输入1-D Tensor，数据类型为float16、float32或float64。
        - **x2** (Tensor) - 输入1-D Tensor，数据类型为float16、float32或float64，输入数据类型需和 `x1` 保持一致。

    返回：
        Tensor，与 `x1` 数据类型相同的输出Tensor。如果 `x1` shape为 :math:`(m,)` ，`x2` shape为 :math:`(n,)` ，则输出的shape为 :math:`(m, n)` 。

    异常：
        - **TypeError** - `x1` 或 `x2` 不是一维Tensor。
        - **TypeError** - 输入的 `x1` 与 `x2` 数据类型不是float16、float32或float64。
        - **TypeError** - 输入的 `x1` 与 `x2` 数据类型不一致。
