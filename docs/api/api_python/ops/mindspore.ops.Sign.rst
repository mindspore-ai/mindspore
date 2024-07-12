mindspore.ops.Sign
===================

.. py:class:: mindspore.ops.Sign

    符号函数，计算输入Tensor元素的执行符号。注意：在输入类型为float64时，该算子反向计算结果为NaN。

    .. math::
        sign(x) = \begin{cases} -1, &if\ x < 0 \cr
        0, &if\ x = 0 \cr
        1, &if\ x > 0\end{cases}

    输入：
        - **x** (Tensor) - 任意维度输入Tensor。

    输出：
        Tensor，shape和数据类型与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。
