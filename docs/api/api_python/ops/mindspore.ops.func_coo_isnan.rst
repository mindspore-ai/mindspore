mindspore.ops.coo_isnan
========================

.. py:function:: mindspore.ops.coo_isnan(x)

    判断COOTensor输入数据每个位置上的值是否是Nan。

    .. math::

        out_i = \begin{cases}
          & \ True,\ \text{ if } x_{i} = \text{Nan} \\
          & \ False,\ \text{ if } x_{i} \ne  \text{Nan}
        \end{cases}

    其中 :math:`Nan` 表示非数字。

    参数：
        - **x** (COOTensor) - 任意维度的COOTensor。

    返回：
        COOTensor，输出的shape与输入相同，数据类型为bool。

    异常：
        - **TypeError** - `x` 不是COOTensor。
