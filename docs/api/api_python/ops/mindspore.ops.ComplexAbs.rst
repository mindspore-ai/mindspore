mindspore.ops.ComplexAbs
=========================

.. py:class:: mindspore.ops.ComplexAbs

    返回输入复数的模。

    .. math::
        y = \sqrt{a^2+b^2}

    输入：
        - **x** (Tensor) - 复数Tensor，格式须为complex64或complex128。

    输出：
        Tensor。如果 `x` 的类型是complex64，则 `y` 的类型是float32；如果 `x` 的类型是complex128，则 `y` 的类型是float64。

    异常：
        - **TypeError** - 输入 `x` 不是Tensor。
        - **TypeError** - 输入 `x` 不是complex64或complex128格式。