mindspore.ops.neg
===================

.. py:function:: mindspore.ops.neg(x)

    计算输入x的相反数并返回。

    .. math::
        out_{i} = - x_{i}

    参数：
        - **x** (Tensor) - Neg的输入Tensor，其秩应在[0, 7]范围内，数据类型为数值型。

    返回：
        Tensor，shape和数据类型与输入相同。

    异常：
        - **TypeError** - `x` 不是Tensor。
