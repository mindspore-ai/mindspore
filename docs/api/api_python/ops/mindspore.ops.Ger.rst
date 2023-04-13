mindspore.ops.Ger
==================

.. py:class:: mindspore.ops.Ger

    计算两个一维Tensor的外积。即输入 `x1` 和输入 `x2` 的外积。如果 `x1` shape为 :math:`(m,)` ，`x2` shape为 :math:`(n,)` ，
    那么输出就是一个shape为 :math:`(m, n)` 的Tensor。

    更多参考详见 :func:`mindspore.ops.ger`。

    输入：
        - **x1** (Tensor) - 输入1-D Tensor。
        - **x2** (Tensor) - 输入1-D Tensor，输入数据类型需和 `x1` 保持一致。

    输出：
        Tensor，与 `x1` 数据类型相同的输出Tensor。如果 `x1` shape为 :math:`(m,)` ， `x2` shape为 :math:`(n,)` ，则输出的shape为 :math:`(m, n)` 。
