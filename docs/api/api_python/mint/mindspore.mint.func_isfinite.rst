mindspore.mint.isfinite
=======================

.. py:function:: mindspore.mint.isfinite(x)

    判断输入数据每个位置上的元素是否是有限数。如果某位置的元素不是 ``NaN`` ， ``-INF`` ， ``INF`` ，则该位置的元素被认为是有限数。

    .. math::

        out_i = \begin{cases}
          & \text{ if } x_{i} = \text{Finite},\ \ True \\
          & \text{ if } x_{i} \ne \text{Finite},\ \ False
        \end{cases}

    参数：
        - **x** (Tensor) - IsFinite的输入。

    返回：
        Tensor，输出的shape与输入相同，数据类型为bool。

    异常：
        - **TypeError** - `x` 不是Tensor。
