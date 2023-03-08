mindspore.ops.swapdims
=======================

.. py:function:: mindspore.ops.swapdims(input, dim0, dim1)

    交换Tensor的两个维度。
    该函数和 :func:`mindspore.ops.swapaxes` 功能一致。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **dim0** (int) - 第一个维度。
        - **dim1** (int) - 第二个维度。

    返回：
        转化后的Tensor，与输入具有相同的数据类型。

    异常：
        - **TypeError** - `input` 不是Tensor类型。
        - **TypeError** - `dim0` 或 `dim1` 不是整数。
        - **ValueError** - `dim0` 或 `dim1` 不在 `[-ndim, ndim-1]` 范围内。