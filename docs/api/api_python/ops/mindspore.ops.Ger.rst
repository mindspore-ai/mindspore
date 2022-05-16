mindspore.ops.Ger
==================

.. py:class:: mindspore.ops.Ger()

    计算两个Tensor的外积，即输入 `x1` 和输入 `x2` 的外积。如果 `x1` shape为 :math:`(m,)` ，`x2` shape为 :math:`(n,)` ，
    那么输出就是一个shape为 :math:`(m, n)` 的Tensor。如果 `x1` shape为 :math:`(*B, m)` ，`x2` shape为 :math:`(*B, n)` ，
    那么输出就是一个shape为 :math:`(*B, m, n)` 的Tensor。

    .. note::
        Ascend 平台暂不支持batch维度输入。即， `x1` 和 `x2` 必须为一维输入Tensor。

    更多参考详见 :func:`mindspore.ops.ger`。
