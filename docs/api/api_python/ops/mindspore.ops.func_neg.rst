mindspore.ops.neg
===================

.. py:function:: mindspore.ops.neg(input)

    计算输入input的相反数并返回。

    .. math::
        out_{i} = - input_{i}

    参数：
        - **input** (Tensor) - Neg的输入Tensor，数据类型为数值型。

    返回：
        Tensor，shape和数据类型与输入相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
