mindspore.Tensor.ger
====================

.. py:method:: mindspore.Tensor.ger(x)

    计算两个Tensor的外积。计算此Tensor 和 `x` 的外积。如果此Tensor shape为 :math:`(m,)` ，`x` shape为 :math:`(n,)` ，
    那么输出就是一个shape为 :math:`(m, n)` 的Tensor。

    .. note::
        Ascend平台暂不支持float64数据格式的输入。

    更多参考详见 :func:`mindspore.ops.ger`。

    参数：
        - **x** (Tensor) - 输入Tensor，数据类型为float16、float32或者float64。

    返回：
        Tensor，是一个与此Tensor相同数据类型的输出矩阵。当此Tensor shape为 :math:`(m,)` ， `x` shape为 :math:`(n,)` ，
        那么输出shape为 :math:`(m, n)` 。