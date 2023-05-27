mindspore.ops.zeta
===================

.. py:function:: mindspore.ops.zeta(input, other)

    逐元素计算Hurwitz zeta的输出。

    .. math::

        \zeta(x, q) = \sum_{k=0}^{\infty} \frac{1}{(k + q)^x}

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Union[Tensor, int, float]) - 输入Tensor。在公式中表示为 :math:`x` ，如果是Tensor，其数据类型必须是float32或float64。
        - **other** (Union[Tensor, int, float]) - 输入Tensor。在公式中表示为 :math:`q` ，如果是Tensor，其数据类型必须和 `input` 相等。

    返回：
        Tensor，Hurwitz zeta的输出。

    异常：
        - **TypeError** - 如果 `input` 和 `other` 均不是Tensor。
        - **TypeError** - `input` 的数据类型不是float32或者float64。
        - **TypeError** - `other` 的数据类型不是float32或者float64。