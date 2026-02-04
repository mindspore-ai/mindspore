mindspore.ops.csr_isnan
========================

.. py:function:: mindspore.ops.csr_isnan(x)

    判断CSRTensor输入数据每个位置上的值是否是NaN。

    .. math::

        out_i = \begin{cases}
          & \ True,\ \text{ if } x_{i} = \text{NaN} \\
          & \ False,\ \text{ if } x_{i} \ne  \text{NaN}
        \end{cases}

    其中 :math:`NaN` 表示非数值（Not a Number）。

    参数：
        - **x** (CSRTensor) - IsNan的输入，任意维度的CSRTensor。

    返回：
        CSRTensor，输出的shape与输入相同，数据类型为bool。

    异常：
        - **TypeError** - `x` 不是CSRTensor。
