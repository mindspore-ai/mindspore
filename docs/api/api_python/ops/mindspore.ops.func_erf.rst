mindspore.ops.erf
=================

.. py:function:: mindspore.ops.erf(input)

    逐元素计算 `input` 的高斯误差函数。

    .. math::

        erf(x)=\frac{2} {\sqrt{\pi}} \int\limits_0^{x} e^{-t^{2}} dt

    参数：
        - **input** (Tensor) - 高斯误差函数的输入Tensor。支持数据类型：

          - Ascend： float16、float32。
          - GPU/CPU： float16、float32、float64。

    返回：
        Tensor，具有与 `input` 相同的数据类型和shape。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 的数据类型既不是float16、float32也不是float64。
