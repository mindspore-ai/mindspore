mindspore.ops.coo_isinf
========================

.. py:function:: mindspore.ops.coo_isinf(x)

    确定输入COOTensor每个位置上的元素是否为正负无穷大。

    .. math::
        out_i = \begin{cases}
        & \text{ if } x_{i} = \text{Inf},\ \ True \\
        & \text{ if } x_{i} \ne \text{Inf},\ \ False
        \end{cases}

    其中 :math:`Inf` 表示正负无穷大。

    参数：
        - **x** (COOTensor) - 任意维度的COOTensor。

    返回：
        COOTensor，shape与输入相同，数据类型为bool。

    异常：
        - **TypeError** - 如果 `x` 不是COOTensor。
