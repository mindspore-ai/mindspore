mindspore.ops.hardsigmoid
=========================

.. py:function:: mindspore.ops.hardsigmoid(input)

    Hard Sigmoid激活函数。按元素计算输出。

    Hard Sigmoid定义为：

    .. math::
        \text{Hardswish}(input) =
        \begin{cases}
        0, & \text{ if } input ≤ -3, \\
        1, & \text{ if } input ≥ +3, \\
        input/6 + 1/2, & \text{ otherwise }
        \end{cases}

    HSigmoid函数图：

    .. image:: ../images/HSigmoid.png
        :align: center

    参数：
        - **input** (Tensor) - 输入Tensor。

    返回：
        Tensor，shape和数据类型与输入 `input` 相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 不是int或者float类型。
