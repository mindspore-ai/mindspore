mindspore.nn.HSigmoid
=============================

.. py:class:: mindspore.nn.HSigmoid

    Hard Sigmoid激活函数。按元素计算输出。

    Hard Sigmoid定义为：

    .. math::
        \text{hsigmoid}(x_{i}) = max(0, min(1, \frac{x_{i} + 3}{6})),

    其中，:math:`x_{i}` 是输入Tensor的一个元素。

    输入：
        - **input_x** (Tensor) - Hard Sigmoid的输入，任意维度的Tensor。

    输出：
        Tensor，数据类型和shape与 `input_x` 的相同。

    异常：
        - **TypeError** - `input_x` 不是tensor。
