mindspore.ops.erfc
==================

.. py:function:: mindspore.ops.erfc(x)

    逐元素计算 `x` 的互补误差函数。

    .. math::

        erfc(x) = 1 - \frac{2} {\sqrt{\pi}} \int\limits_0^{x} e^{-t^{2}} dt

    参数：
        - **x** (Tensor) - 互补误差函数的输入Tensor。维度必须小于8，数据类型必须为float16、float32或float64。

    返回：
        Tensor，具有与 `x` 相同的数据类型和shape。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `x` 的数据类型既不是float16、float32或float64。
