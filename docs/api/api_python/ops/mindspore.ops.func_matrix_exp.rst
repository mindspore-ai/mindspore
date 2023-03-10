mindspore.ops.matrix_exp
========================

.. py:function:: mindspore.ops.matrix_exp(input)

    计算单个或一批方阵的矩阵指数。

    .. math::

        matrix\_exp(x) = \sum_{k=0}^{\infty} \frac{1}{k !} x^{k} \in \mathbb{K}^{n \times n}，其中 :math:`input` 即输入 `input`。

    参数：
        - **input** (Tensor) - 输入Tensor，shape为 :math:`(*, n, n)` ，其中 `*` 表示0或更多的batch维。
          支持数据类型：float16、float32、float64、complex64、complex128。

    返回：
        Tensor，其shape和数据类型均与 `input` 相同。

    异常：
        - **TypeError** -  `input` 不为Tensor。
        - **TypeError** -  `input` 的dtype不属于以下类型：float16、float32、float64、complex64、complex128。
        - **ValueError** - `input` 的秩小于2。
        - **ValueError** - `input` 的最后两维不相等。
