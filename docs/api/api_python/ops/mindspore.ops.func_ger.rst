mindspore.ops.ger
========================

.. py:function:: mindspore.ops.ger(x1, x2)

    计算两个Tensor的外积，即输入 `x1` 和输入 `x2` 的外积。如果 `x1` shape为 :math:`(m,)` ，`x2` shape为 :math:`(n,)` ，
    那么输出就是一个shape为 :math:`(m, n)` 的Tensor。如果 `x1` shape为 :math:`(*B, m)` ，`x2` shape为 :math:`(*B, n)` ，
    那么输出就是一个shape为 :math:`(*B, m, n)` 的Tensor。

    .. note::
        Ascend 平台暂不支持batch维度输入。即， `x1` 和 `x2` 必须为一维输入Tensor。

    **参数：**

    - **x1** (Tensor) - 输入Tensor，数据类型为float16或float32。
    - **x2** (Tensor) - 输入Tensor，数据类型为float16或float32。

    **返回：**

    Tensor，是一个与 `x1` 相同数据类型的输出矩阵。当 `x1` shape为 :math:`(*B, m)` ， `x2` shape为 :math:`(*B, n)` ，那么输出shape为 :math:`(*B, m, n)` 。

    **异常：**

    - **TypeError** - `x1` 或 `x2` 不是Tensor。
    - **RuntimeError** - 输入的 `x1` 与 `x2` 数据类型不是float16或float32。
    - **RuntimeError** - `x1` 与 `x2` batch维 :math:`(*B)` 不相同。
