mindspore.ops.IsInf
====================

.. py:class:: mindspore.ops.IsInf

    确定输入Tensor每个位置上的元素是否为无穷大或无穷小。

    .. math::
        out_i = \begin{cases}
        & \text{ if } x_{i} = \text{Inf},\ \ True \\
        & \text{ if } x_{i} \ne \text{Inf},\ \ False
        \end{cases}

    其中 :math:`Inf` 表示不是一个数字。

    **输入：**

    - **x** (Tensor) - IsInf的输入，任意维度的Tensor。

    **输出：**

    Tensor，shape与相同的输入，数据的类型为bool。

    **异常：**

    - **TypeError** - 如果 `x` 不是Tensor。