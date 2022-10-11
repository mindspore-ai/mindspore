mindspore.ops.hardswish
=======================

.. py:function:: mindspore.ops.hardswish(x)

    Hard Swish激活函数。

    对输入的每个元素计算Hard Swish。输入是一个张量，具有任何有效的shape。

    Hard Swish定义如下：

    .. math::
        \text{hardswish}(x_{i}) = x_{i} * \frac{ReLU6(x_{i} + 3)}{6}

    其中， :math:`x_i` 是输入的元素。

    参数：
        - **x** (Tensor) - 用于计算Hard Swish的Tensor。数据类型必须是float16或float32。

    返回：
        Tensor，shape和数据类型与输入相同。

    异常：
        - **TypeError** - `x` 不是一个Tensor。
        - **TypeError** - `x` 的数据类型既不是float16也不是float32。