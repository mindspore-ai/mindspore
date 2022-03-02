mindspore.ops.Round
====================

.. py:class:: mindspore.ops.Round

    对输入数据进行四舍五入到最接近的整数数值。

    .. math::
        out_i \approx x_i

    **输入：**

    - **x** (Tensor) - 输入Tensor。

    **输出：**

    Tensor，shape和数据类型与输入 `x` 相同。

    **异常：**

    - **TypeError** - `x` 不是Tensor。