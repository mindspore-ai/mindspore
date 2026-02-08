mindspore.ops.coo_isfinite
===========================

.. py:function:: mindspore.ops.coo_isfinite(x)

    判断COOTensor输入数据每个位置上的值是否是有限数。

    .. math::

        out_i = \begin{cases}
          & \text{ if } x_{i} = \text{Finite},\ \ True\  \\
          & \text{ if } x_{i} \ne \text{Finite},\ \ False
        \end{cases}

    参数：
        - **x** (COOTensor) - 任意维度的COOTensor。

    返回：
        COOTensor，输出的shape与输入相同，数据类型为bool。

    异常：
        - **TypeError** - `x` 不是COOTensor。
