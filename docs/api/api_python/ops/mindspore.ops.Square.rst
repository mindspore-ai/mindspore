mindspore.ops.Square
=====================

.. py:class:: mindspore.ops.Square

    计算输入Tensor的平方。

    .. math::
        out_{i} = (x_{i})^2

    输入：
        - **x** (Tensor) - Square的输入，其数据类型为数值型，shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。其秩应小于8。

    输出：
        Tensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。