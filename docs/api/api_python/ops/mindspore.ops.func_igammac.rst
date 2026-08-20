mindspore.ops.igammac
=====================

.. py:function:: mindspore.ops.igammac(input, other)

    计算正则化的上层不完全伽马函数。

    将 `input` 记作 `a` ， `other` 记作 `x` ，则正则化的上层不完全伽马函数可以表示为：

    .. math::
        Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x)

    其中，

    .. math::
        Gamma(a, x) = \int_{x}^{\infty} t^{a-1} \exp(-t) dt

    表示上层不完全伽马函数。

    :math:`P(a, x)` 表示正则化的下层不完全伽马函数。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 第一个输入tensor。
        - **other** (Tensor) - 第二个输入tensor。

    返回：
        Tensor，数据类型与 `input` 和 `other` 相同。
