mindspore.ops.digamma
=====================

.. py:function:: mindspore.ops.digamma(input)

    计算对数gamma函数在输入上的导数。

    .. math::
        P(x) = \frac{d}{dx}(\ln (\Gamma(x)))

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入Tensor，数据类型是float16，float32或float64。

    返回：
        Tensor，数据类型和 `input` 一样。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 的数据类型不是float16，float32或float64。
