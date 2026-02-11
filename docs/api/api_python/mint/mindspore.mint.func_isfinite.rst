mindspore.mint.isfinite
=======================

.. py:function:: mindspore.mint.isfinite(input)

    逐元素返回输入tensor中元素是否是有限数。
    有限数定义：非 ``NaN`` 、 ``-INF`` 、 ``INF`` 的元素。

    .. math::

        out_i = \begin{cases}
          & \text{ if } input_{i} = \text{Finite},\ \ True \\
          & \text{ if } input_{i} \ne \text{Finite},\ \ False
        \end{cases}

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        Bool型Tensor
