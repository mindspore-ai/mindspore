mindspore.ops.csr_tan
======================

.. py:function:: mindspore.ops.csr_tan(x)

    逐元素计算CSRTensor的正切值。

    .. math::
        out_i = \tan(x_i)

    参数：
        - **x** (CSRTensor) - Tan的输入，任意维度的CSRTensor。

    返回：
        CSRTensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是CSRTensor。
