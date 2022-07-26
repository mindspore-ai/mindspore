mindspore.ops.inv
=================

.. py:function:: mindspore.ops.inv(x)

    逐元素计算输入Tensor的倒数。

    .. math::
        out_i = \frac{1}{x_{i} }

    参数：
        - **x** (Tensor) - 任意维度的Tensor。其数据类型为float16、float32或int32。

    返回：
        Tensor，shape和类型与输入相同。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `x` 的数据类型不为float16、float32或int32。
