mindspore.ops.igamma
====================

.. py:function:: mindspore.ops.igamma(input, other)

    计算正规化的下层不完全伽马函数。
    如果我们将 `input` 比作 `a` ， `other` 比作 `x` ，则正规化的下层不完全伽马函数可以表示成：

    .. math::
        P(a, x) = Gamma(a, x) / Gamma(a) = 1 - Q(a, x)

    其中，

    .. math::
        Gamma(a, x) = \int_0^x t^{a-1} \exp^{-t} dt

    为下层不完全伽马函数。

    :math:`Q(a, x)` 则为正规化的上层完全伽马函数。   
 
    .. warning::
        这是一个实验原型，可以更改和/或删除。

    参数：
        - **input** (Tensor) - 输入Tensor，数据类型为float32或者float64。
        - **other** (Tensor) - 输入Tensor，数据类型为float32或者float64，与 `input` 保持一致。

    返回：
        Tensor，数据类型与 `input` 和 `other` 相同。

    异常：
        - **TypeError** - 如果 `input` 或者 `other` 不是Tensor。
        - **TypeError** - 如果 `other` 的数据类型不是float32或者float64。
        - **TypeError** - 如果 `other` 的数据类型与 `input` 不相同。
        - **ValueError** - 如果 `input` 不能广播成shape与 `other` 相同的Tensor。
