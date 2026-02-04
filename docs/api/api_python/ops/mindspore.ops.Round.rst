mindspore.ops.Round
===================

.. py:class:: mindspore.ops.Round

    对输入数据进行四舍五入，得到最接近的整数数值。

    .. math::
        out_i \approx input_i

    .. note::
        Ascend平台支持的输入数据类型包括bfloat16（Atlas训练系列产品不支持）、float16、float32、float64、int32、int64。

    输入：
        - **input** (Tensor) - 输入Tensor。
        - **decimals** (int, 可选) - 要舍入到的小数位数（默认值： ``0``）。如果为负数，则指定小数点左侧的位数。支持输入单元素Tensor转换为int。 `input` 类型为int32或int64时， `decimals` 参数值必须为0。

    输出：
        Tensor，shape和数据类型与输入 `input` 相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **RuntimeError** - `input` 为int32或int64类型时， `decimals` 值不为0。