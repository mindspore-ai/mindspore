mindspore.ops.Ceil
===================

.. py:class:: mindspore.ops.Ceil

    向上取整函数。

    .. math::
        out_i = \lceil x_i \rceil = \lfloor x_i \rfloor + 1

    **输入：**

    - **x** (Tensor) - Ceil的输入，任意维度的Tensor，秩应小于8。其数据类型为float16或float32。

    **输出：**

    Tensor，shape与 `x` 相同。

    **异常：**

    - **TypeError** - `x` 的不是Tensor。
    - **TypeError** - `x` 的数据类型既不是float16也不是float32。