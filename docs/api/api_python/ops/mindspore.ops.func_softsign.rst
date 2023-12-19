mindspore.ops.softsign
======================

.. py:function:: mindspore.ops.softsign(x)

    SoftSign激活函数。

    SoftSign函数定义为：

    .. math::
        \text{SoftSign}(x) = \frac{x}{1 + |x|}

    Softsign函数图：

    .. image:: ../images/Softsign.png
        :align: center

    参数：
        - **x** (Tensor) - shape为 :math:`(N, *)` 的Tensor，其中 :math:`*` 表示任意个数的维度。它的数据类型必须为float16或float32。

    返回：
        Tensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `x` 的数据类型既不是float16也不是float32。
