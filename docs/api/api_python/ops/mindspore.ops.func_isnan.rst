mindspore.ops.isnan
====================

.. py:function:: mindspore.ops.isnan(x)

    判断输入数据每个位置上的值是否是Nan。

    .. math::

        out_i = \begin{cases}
          & \text{ if } x_{i} = \text{Nan},\ \ True \\
          & \text{ if } x_{i} \ne  \text{Nan},\ \ False
        \end{cases}

    其中 :math:`Nan` 表示的不是number。

    参数：
        - **x** (Tensor) - IsNan的输入，任意维度的Tensor。

    返回：
        Tensor，输出的shape与输入相同，数据类型为bool。

    异常：
        - **TypeError** - `x` 不是Tensor。
