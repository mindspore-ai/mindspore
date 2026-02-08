mindspore.ops.csr_neg
======================

.. py:function:: mindspore.ops.csr_neg(x)

    计算输入CSRTensor的相反数并返回。

    .. math::
        out_{i} = - x_{i}

    参数：
        - **x** (CSRTensor) - Neg的输入，其数据类型为数值型。

    返回：
        CSRTensor，shape和类型与输入相同。

    异常：
        - **TypeError** - `x` 不是CSRTensor。