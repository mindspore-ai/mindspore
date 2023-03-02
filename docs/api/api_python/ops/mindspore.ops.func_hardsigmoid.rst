mindspore.ops.hardsigmoid
=========================

.. py:function:: mindspore.ops.hardsigmoid(input_x)

    Hard Sigmoid激活函数。按元素计算输出。

    Hard Sigmoid定义为：

    .. math::
        \text{hsigmoid}(x_{i}) = max(0, min(1, \frac{x_{i} + 3}{6})),

    其中，:math:`x_i` 是输入Tensor的一个元素。

    参数：
        - **input_x** (Tensor) - Hard Sigmoid的输入，任意维度的Tensor，数据类型为float16、float32或float64。

    返回：
        Tensor，shape和数据类型与输入 `input_x` 相同。

    异常：
        - **TypeError** - `input_x` 不是Tensor。
        - **TypeError** - `input_x` 的dtype不是float16、float32或float64。