mindspore.ops.csr_sqrt
=======================

.. py:function:: mindspore.ops.csr_sqrt(x)

    逐元素返回当前CSRTensor的平方根。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    .. math::
        out_{i} = \sqrt{x_{i}}

    参数：
        - **x** (CSRTensor) - 输入CSRTensor，数据类型为Number。

    返回：
        CSRTensor，具有与 `x` 相同的shape和数据类型。

    异常：
        - **TypeError** - 如果 `x` 不是CSRTensor。
