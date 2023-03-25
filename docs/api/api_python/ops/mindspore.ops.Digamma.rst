mindspore.ops.Digamma
======================

.. py:class:: mindspore.ops.Digamma

    计算输入的lgamma函数的导数。

    .. math::
        P(x) = grad(ln(gamma(x)))

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    输入：
        - **x** (Tensor) - 输入Tensor。数据类型为 `float16` 、 `float32` 或者 `float64` 。

    输出：
        Tensor, 和输入 `x` 具有相同的数据类型。

    异常：
        - **TypeError** - 如果输入 `x` 不是Tensor。
        - **TypeError** - 输入输入 `x` 的数据类型不是 `float16` 、 `float32` 或者 `float64` 。
