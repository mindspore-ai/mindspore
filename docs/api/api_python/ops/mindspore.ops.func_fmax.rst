mindspore.ops.fmax
==================

.. py:function:: mindspore.ops.fmax(input, other)

    逐元素计算输入Tensor的最大值。

    .. math::
        output_i = max(x1_i, x2_i)

    .. note::
        - 输入 `input` 和 `other` 遵循隐式转换法则使数据类型一致。
        - 输入 `input` 和 `other` 的shape必须能相互广播。
        - 如果其中一个比较值是NaN，则返回另一个比较值。

    参数：
        - **input** (Tensor) - 第一个输入Tensor，支持的数据类型有： float16、float32、 float64、 int32、 int64。
        - **other** (Tensor) - 第二个输入Tensor，支持的数据类型有： float16、 float32、 float64、 int32、 int64。

    返回：
        Tensor。其shape与两个输入广播之后的shape相同，数据类型为隐式转换后精度较高的数据类型。

    异常：
        - **TypeError** - `input` 或 `other` 不是Tensor。
        - **TypeError** - `input` 或 `other` 的数据类型不是以下数据类型之一：float16、 float32、 float64、 int32、 int64。
        - **ValueError** - `input` 和 `other` 的shape不能广播。
