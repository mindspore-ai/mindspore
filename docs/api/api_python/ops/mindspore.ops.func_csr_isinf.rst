mindspore.ops.csr_isinf
========================

.. py:function:: mindspore.ops.csr_isinf(x)

    确定输入CSRTensor每个位置上的元素是否为无穷大或无穷小。

    .. math::
        out_i = \begin{cases}
        & \text{ if } x_{i} = \text{Inf},\ \ True \\
        & \text{ if } x_{i} \ne \text{Inf},\ \ False
        \end{cases}

    其中 :math:`Inf` 表示不是一个数字。

    参数：
        - **x** (CSRTensor) - IsInf的输入。

    返回：
        CSRTensor，shape与相同的输入，数据的类型为bool。

    异常：
        - **TypeError** - 如果 `x` 不是CSRTensor。
