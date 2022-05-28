mindspore.ops.tan
===================

.. py:function:: mindspore.ops.tan(x)

    计算输入元素的正切值。

    .. math::
        out_i = tan(x_i)

    **参数：**

    - **x** (Tensor) - Tan的输入，任意维度的Tensor，其数据类型为float16或float32。

    **返回：**

    Tensor，数据类型和shape与 `x` 相同。

    **异常：**

    - **TypeError** - `x` 的数据类型既不是float16也不是float32。
    - **TypeError** - `x` 不是Tensor。
