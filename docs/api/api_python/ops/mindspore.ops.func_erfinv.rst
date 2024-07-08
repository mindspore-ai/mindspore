mindspore.ops.erfinv
====================

.. py:function:: mindspore.ops.erfinv(input)

    计算输入的逆误差函数。逆误差函数在 `(-1, 1)` 范围内定义为：

    .. math::
        erfinv(erf(x)) = x

    其中 :math:`x` 代表输入Tensor。

    参数：
        - **input** (Tensor) - 输入Tensor。支持数据类型：

          - Ascend： float16、float32、int8、int16、int32、int64、uint8、bool。
          - GPU/CPU： float16、float32、float64。

    返回：
        Tensor。当输入为 int8、int16、int32、int64、uint8、bool 时，返回值类型为float32。
        否则，返回值类型与输入类型相同。

    异常：
        - **TypeError** - `input` 的数据类型不是如下类型：

          - Ascend: float16、float32、int8、int16、int32、int64、uint8、bool。
          - CPU/GPU: float16、float32、float64。
