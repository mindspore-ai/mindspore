mindspore.ops.hardsigmoid
=========================

.. py:function:: mindspore.ops.hardsigmoid(input)

    Hard Sigmoid激活函数。按元素计算输出。

    Hard Sigmoid定义为：

    .. math::
        \text{hsigmoid}(x_{i}) = \max(0, \min(1, \frac{x_{i} + 3}{6}))

    其中，:math:`x_i` 是输入Tensor的一个元素。

    参数：
        - **input** (Tensor) - 输入Tensor。

    返回：
        Tensor，shape和数据类型与输入 `input` 相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 的dtype不是int或者float类型。
