mindspore.ops.swapaxes
=======================

.. py:function:: mindspore.ops.swapaxes(x, axis0, axis1)

    交换Tensor的两个维度。

    参数：
        - **x** (Tensor) - 输入Tensor。
        - **axis0** (int) - 第一个维度。
        - **axis1** (int) - 第二个维度。

    返回：
        转化后的Tensor，与输入具有相同的数据类型。

    异常：
        - **TypeError** - `x` 不是Tensor类型。
        - **TypeError** - `axis0` 或 `axis1` 不是整数。
        - **ValueError** - `axis0` 或 `axis1` 不在 `[-ndim, ndim-1]` 范围内。