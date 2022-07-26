mindspore.ops.Reciprocal
=========================

.. py:class:: mindspore.ops.Reciprocal

    返回输入Tensor的倒数。

    .. math::
        out_{i} =  \frac{1}{x_{i}}

    输入：
        - **x** (Tensor) - Reciprocal的输入。shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。

    输出：
        Tensor，shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。