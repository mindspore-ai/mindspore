mindspore.ops.csr_sqrt
=======================

.. py:function:: mindspore.ops.csr_sqrt(x)

    逐元素返回当前CSRTensor的平方根。

    .. math::
        out_{i} = \sqrt{x_{i}}

    参数：
        - **x** (CSRTensor) - 输入CSRTensor，数据类型为number.Number。

    返回：
        CSRTensor，具有与 `x` 相同的shape。

    异常：
        - **TypeError** - 如果 `x` 不是CSRTensor。

