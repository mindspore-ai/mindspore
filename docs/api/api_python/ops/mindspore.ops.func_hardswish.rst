mindspore.ops.hardswish
=======================

.. py:function:: mindspore.ops.hardswish(x)

    逐元素计算Hard Swish。输入是一个Tensor，具有任何有效的shape。

    Hard Swish定义如下：

    .. math::
        \text{hswish}(x_{i}) = x_{i} * \frac{ReLU6(x_{i} + 3)}{6}

    其中， :math:`x_i` 是输入的元素。

    参数：
        - **x** (Tensor) - 用于计算Hard Swish的Tensor。

    返回：
        Tensor，shape和数据类型与输入相同。

    异常：
        - **TypeError** - `x` 不是一个Tensor。
        - **TypeError** - `x` 的数据类型int或者float。