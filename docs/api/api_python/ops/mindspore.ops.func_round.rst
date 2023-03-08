mindspore.ops.round
====================

.. py:function:: mindspore.ops.round(input)

    对输入数据进行四舍五入到最接近的整数数值。

    .. math::
        out_i \approx input_i

    参数：
        - **input** (Tensor) - 输入Tensor。

    返回：
        Tensor，shape和数据类型与输入 `input` 相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
