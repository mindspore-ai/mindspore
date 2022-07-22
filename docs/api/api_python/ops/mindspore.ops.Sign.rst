mindspore.ops.Sign
===================

.. py:class:: mindspore.ops.Sign

    符号函数，计算输入Tensor元素的执行符号。

    .. math::
        sign(x) = \begin{cases} -1, &if\ x < 0 \cr
        0, &if\ x = 0 \cr
        1, &if\ x > 0\end{cases}

    输入：
        - **x** (Tensor) - Sign的输入，shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。

    输出：
        Tensor，shape和数据类型与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。
