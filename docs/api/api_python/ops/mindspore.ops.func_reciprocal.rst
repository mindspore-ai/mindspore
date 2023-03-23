mindspore.ops.reciprocal
=========================

.. py:function:: mindspore.ops.reciprocal(input)

    返回输入Tensor每个元素的倒数。

    .. math::
        out_{i} =  \frac{1}{x_{i}}

    参数：
        - **input** (Tensor) - 输入Tensor。shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。

    返回：
        Tensor，shape与 `input` 相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
