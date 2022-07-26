mindspore.ops.round
====================

.. py:function:: mindspore.ops.round(x)

    对输入数据进行四舍五入到最接近的整数数值。

    .. math::
        out_i \approx x_i

    参数：
        - **x** (Tensor) - 输入Tensor。

    返回：
        Tensor，shape和数据类型与输入 `x` 相同。

    异常：
        - **TypeError** - `x` 不是Tensor。
