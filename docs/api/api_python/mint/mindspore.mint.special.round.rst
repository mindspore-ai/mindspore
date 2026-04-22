mindspore.mint.special.round
============================

.. py:function:: mindspore.mint.special.round(input)

    对输入数据进行四舍五入到最接近的整数数值。

    .. math::
        out_i \approx input_i

    .. note::
        Ascend平台支持的输入数据类型包括bfloat16、float16、float32、float64、int32、int64。

    参数：
        - **input** (Tensor) - 输入Tensor。

    返回：
        Tensor，shape和数据类型与输入 `input` 相同。

    异常：
        - **TypeError** - `input` 不是Tensor。