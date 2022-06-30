mindspore.ops.ger
=================

.. py:function:: mindspore.ops.ger(x1, x2)

    计算两个一维Tensor的外积，即输入 `x1` 和输入 `x2` 的外积。如果 `x1` shape为 :math:`(m,)` ，`x2` shape为 :math:`(n,)` ，
    那么输出就是一个shape为 :math:`(m, n)` 的Tensor。

    .. note::
        Ascend不支持float64数据格式的输入。

    **参数：**

    - **x1** (Tensor) - 输入1-D Tensor，数据类型为float16、float32或float64。
    - **x2** (Tensor) - 输入1-D Tensor，数据类型为float16、float32或float64， 输入数据类型需和 `x1` 保持一致。

    **返回：**

    Tensor，是一个与 `x1` 相同数据类型的输出矩阵。如果 `x1` shape为 :math:`(m,)` ，`x2` shape为 :math:`(n,)` ，那么输出就是一个shape为 :math:`(m, n)` 的Tensor。

    **异常：**

    - **TypeError** - `x1` 或 `x2` 不是一维Tensor。
    - **TypeError** - 输入的 `x1` 与 `x2` 数据类型不是float16、float32或float64。
    - **TypeError** - 输入的 `x1` 与 `x2` 数据类型不一致。
