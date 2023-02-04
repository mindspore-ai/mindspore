mindspore.ops.fmin
==================

.. py:function:: mindspore.ops.fmin(x1, x2)

    逐元素计算输入Tensor的最小值。

    .. math::
        output_i = min(x1_i, x2_i)

    .. note::
        - 输入 `x1` 和 `x2` 遵循隐式转换法则使数据类型一致。
        - 输入 `x1` 和 `x2` 的shape必须能相互广播。
        - 如果其中一个比较值是NaN，则返回另一个比较值。

    参数：
        - **x1** (Tensor) - 第一个输入Tensor，支持的数据类型有： float16、float32、 float64、 int32、 int64。
        - **x2** (Tensor) - 第二个输入Tensor，支持的数据类型有： float16、 float32、 float64、 int32、 int64。

    返回：
        Tensor。其shape与两个输入广播之后的shape相同，数据类型为隐式转换后精度较高的数据类型。

    异常：
        - **TypeError** - `x1` 或 `x2` 不是Tensor。
        - **TypeError** - `x1` 或 `x2` 的数据类型不是以下数据类型之一：float16、 float32、 float64、 int32、 int64。
        - **ValueError** - `x1` 和 `x2` 的shape不能广播。
