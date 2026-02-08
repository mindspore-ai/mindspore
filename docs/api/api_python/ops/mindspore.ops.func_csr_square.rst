mindspore.ops.csr_square
=========================

.. py:function:: mindspore.ops.csr_square(x)

    逐元素返回CSRTensor的平方。

    .. math::
        out_{i} = (x_{i})^2

    参数：
        - **x** (CSRTensor) - 输入CSRTensor，数据类型为数值类型。

    返回：
        CSRTensor，具有与当前CSRTensor相同的数据类型和shape。

    异常：
        - **TypeError** - `x` 不是CSRTensor。
