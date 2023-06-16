mindspore.ops.isinf
===================

.. py:function:: mindspore.ops.isinf(input)

    确定输入Tensor每个位置上的元素是否为无穷大或无穷小。

    .. math::

        out_i = \begin{cases}
          & \ True,\ \text{ if } x_{i} = \text{Inf} \\
          & \ False,\ \text{ if } x_{i} \ne  \text{Inf}
        \end{cases}

    其中 :math:`Inf` 表示不是一个数字。

    参数：
        - **input** (Tensor) - 输入Tensor。

    返回：
        Tensor，shape与相同的输入，数据的类型为bool。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
