mindspore.ops.digamma
=====================

.. py:function:: mindspore.ops.digamma(input)

    计算gamma对数函数在输入上的梯度。

    .. math::
        P(x) = grad(In(gamma(x)))

    .. warning::
        这是一个实验性接口，后续可能删除或修改。

    参数：
        - **input** (Tensor) - 输入Tensor，数据类型是float16，float32或float64。

    返回：
        Tensor，数据类型和 `input` 一样。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 的数据类型不是float16，float32或float64。
